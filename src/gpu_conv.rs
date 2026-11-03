//! CUDA convolutional encoder/decoders. FP32 GEMM, FP64 normalization statistics.
use crate::{gpu, weights::Weights};
use burn::backend::{backend_extension, tensor::FloatTensor, Backend, Dispatch};
use burn::tensor::{Device, Tensor, TensorData};
use burn_cubecl::{
    kernel::into_contiguous,
    ops::numeric::{empty_device_contiguous_dtype, empty_device_dtype},
    tensor::CubeTensor,
    CubeBackend,
};
use cubecl::{calculate_cube_count_elemwise, prelude::CubeDim};
use eyre::Result;

#[backend_extension(Cube)]
trait ConvOps: Backend {
    #[allow(clippy::too_many_arguments)]
    fn columns(
        x: FloatTensor<Self>,
        time: usize,
        freq: usize,
        kt: usize,
        kf: usize,
        dilation: usize,
        pt: usize,
        pf: usize,
        st: usize,
        sf: usize,
        fo: usize,
        first: usize,
        count: usize,
    ) -> FloatTensor<Self>;
    fn strict_mm(a: FloatTensor<Self>, b: FloatTensor<Self>) -> FloatTensor<Self>;
    fn norm_act(
        x: FloatTensor<Self>,
        w: FloatTensor<Self>,
        b: FloatTensor<Self>,
        p: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

mod kernels {
    use cubecl::prelude::*;

    #[cube(launch)]
    pub fn columns(
        x: &Tensor<f32>,
        y: &mut Tensor<f32>,
        time: u32,
        freq: u32,
        kt: u32,
        kf: u32,
        dilation: u32,
        pt: u32,
        pf: u32,
        st: u32,
        sf: u32,
        fo: u32,
        first: u32,
        count: u32,
    ) {
        let at = ABSOLUTE_POS;
        if at >= y.len() {
            terminate!();
        }
        let pos = first as usize + at % count as usize;
        let k = at / count as usize;
        let b = k % kf as usize;
        let a = k / kf as usize % kt as usize;
        let ci = k / (kt * kf) as usize;
        let ti = ((pos / fo as usize) * st as usize + a * dilation as usize) as i32 - pt as i32;
        let fi = ((pos % fo as usize) * sf as usize + b) as i32 - pf as i32;
        let mut v = 0.0f32;
        if ti >= 0 && ti < time as i32 && fi >= 0 && fi < freq as i32 {
            v = x[(ci * time as usize + ti as usize) * freq as usize + fi as usize];
        }
        y[at] = v;
    }

    // Two FP64 moments per 1024-value piece; never materialize an FP64 plane.
    #[cube(launch)]
    pub fn moments(x: &Tensor<f32>, partial: &mut Tensor<f64>, plane: u32, pieces: u32) {
        let at = ABSOLUTE_POS;
        if at >= partial.len() / 2 {
            terminate!();
        }
        let c = at / pieces as usize;
        let piece = at % pieces as usize;
        let start = piece * 1024usize;
        let end = (start + 1024usize).min(plane as usize);
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for i in start..end {
            let v = f64::cast_from(x[c * plane as usize + i]);
            sum += v;
            sq += v * v;
        }
        partial[at * 2] = sum;
        partial[at * 2 + 1] = sq;
    }

    #[cube(launch)]
    pub fn coefficients(
        partial: &Tensor<f64>,
        w: &Tensor<f32>,
        b: &Tensor<f32>,
        p: &Tensor<f32>,
        coeff: &mut Tensor<f32>,
        plane: u32,
        pieces: u32,
    ) {
        let c = ABSOLUTE_POS;
        if c >= w.len() {
            terminate!();
        }
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for i in 0..pieces as usize {
            sum += partial[(c * pieces as usize + i) * 2];
            sq += partial[(c * pieces as usize + i) * 2 + 1];
        }
        let mean = sum / f64::cast_from(plane);
        let variance = (sq / f64::cast_from(plane) - mean * mean).max(0.0f64);
        let scale = f64::cast_from(w[c]) / (variance + 0.00001f64).sqrt();
        let shift = f64::cast_from(b[c]) - mean * scale;
        coeff[c * 3] = f32::cast_from(scale);
        coeff[c * 3 + 1] = f32::cast_from(shift);
        coeff[c * 3 + 2] = p[c];
    }

    #[cube(launch)]
    pub fn scale(x: &Tensor<f32>, coeff: &Tensor<f32>, y: &mut Tensor<f32>, plane: u32) {
        let at = ABSOLUTE_POS;
        if at >= y.len() {
            terminate!();
        }
        y[at] = x[at] * coeff[(at / plane as usize) * 3];
    }

    #[cube(launch)]
    pub fn apply(x: &Tensor<f32>, coeff: &Tensor<f32>, y: &mut Tensor<f32>, plane: u32) {
        let at = ABSOLUTE_POS;
        if at >= y.len() {
            terminate!();
        }
        let c = at / plane as usize;
        let v = x[at] + coeff[c * 3 + 1];
        let mut out = v;
        if v <= 0.0f32 {
            out = v * coeff[c * 3 + 2];
        }
        y[at] = out;
    }
}

fn empty(x: &CubeTensor, shape: impl Into<burn::tensor::Shape>) -> CubeTensor {
    // These kernels index flat logical rows. The generic allocator may pad
    // their physical row strides, especially at odd STFT dimensions.
    empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), shape.into(), x.dtype)
}

fn grid(x: &CubeTensor, n: usize) -> cubecl::CubeCount {
    calculate_cube_count_elemwise(&x.client, n, CubeDim::new_1d(256))
}

impl ConvOps for CubeBackend {
    fn columns(
        x: CubeTensor,
        time: usize,
        freq: usize,
        kt: usize,
        kf: usize,
        dilation: usize,
        pt: usize,
        pf: usize,
        st: usize,
        sf: usize,
        fo: usize,
        first: usize,
        count: usize,
    ) -> CubeTensor {
        let x = into_contiguous(x);
        let k = x.meta.shape()[0] * kt * kf;
        let y = empty(&x, [k, count]);
        kernels::columns::launch(
            &x.client,
            grid(&x, k * count),
            CubeDim::new_1d(256),
            x.clone().into_tensor_arg(),
            y.clone().into_tensor_arg(),
            time as u32,
            freq as u32,
            kt as u32,
            kf as u32,
            dilation as u32,
            pt as u32,
            pf as u32,
            st as u32,
            sf as u32,
            fo as u32,
            first as u32,
            count as u32,
        );
        y
    }

    fn strict_mm(a: CubeTensor, b: CubeTensor) -> CubeTensor {
        use cubek::{
            matmul::{
                definition::{MatmulElems, MatmulGlobalElems},
                multi_level::Strategy,
            },
            std::InputBinding,
        };
        let out = empty(&a, [a.meta.shape()[0], b.meta.shape()[1]]);
        let dtype = burn::backend::cubecl::dtype_to_storage_type(a.dtype);
        let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
            lhs: dtype,
            rhs: dtype,
            out: dtype,
        });
        // SimpleUnit keeps FP32 products/accumulation. Auto can silently choose TF32.
        cubek::matmul::launch::launch_ref(
            &Strategy::SimpleUnit(Default::default()).into(),
            &out.client,
            InputBinding::new(a.binding(), dtype),
            InputBinding::new(b.binding(), dtype),
            out.clone().binding(),
            &mut dtypes,
        )
        .expect("FP32 CUDA GEMM");
        out
    }

    fn norm_act(x: CubeTensor, w: CubeTensor, b: CubeTensor, p: CubeTensor) -> CubeTensor {
        let x = into_contiguous(x);
        let channels = x.meta.shape()[0];
        let plane = x.meta.shape()[1];
        let pieces = plane.div_ceil(1024);
        let partial = empty_device_dtype(
            x.client.clone(),
            x.device.clone(),
            [channels * pieces * 2].into(),
            burn::tensor::DType::F64,
        );
        let coeff = empty(&x, [channels * 3]);
        let scaled = empty(&x, [channels, plane]);
        let y = empty(&x, [channels, plane]);
        kernels::moments::launch(
            &x.client,
            grid(&x, channels * pieces),
            CubeDim::new_1d(256),
            x.clone().into_tensor_arg(),
            partial.clone().into_tensor_arg(),
            plane as u32,
            pieces as u32,
        );
        kernels::coefficients::launch(
            &x.client,
            grid(&x, channels),
            CubeDim::new_1d(256),
            partial.into_tensor_arg(),
            w.into_tensor_arg(),
            b.into_tensor_arg(),
            p.into_tensor_arg(),
            coeff.clone().into_tensor_arg(),
            plane as u32,
            pieces as u32,
        );
        // The CPU rounds multiplication before addition. A fused GPU FMA
        // perturbs the encoder enough to change later TF32 rounding decisions.
        kernels::scale::launch(
            &x.client,
            grid(&x, channels * plane),
            CubeDim::new_1d(256),
            x.clone().into_tensor_arg(),
            coeff.clone().into_tensor_arg(),
            scaled.clone().into_tensor_arg(),
            plane as u32,
        );
        kernels::apply::launch(
            &x.client,
            grid(&x, channels * plane),
            CubeDim::new_1d(256),
            scaled.into_tensor_arg(),
            coeff.into_tensor_arg(),
            y.clone().into_tensor_arg(),
            plane as u32,
        );
        y
    }
}

fn tensor<const D: usize>(w: &Weights, name: &str, device: &Device) -> Result<Tensor<D>> {
    let shape: [usize; D] = w
        .shape(name)?
        .try_into()
        .map_err(|_| eyre::eyre!("invalid shape for {name}"))?;
    Ok(Tensor::from_data(
        TensorData::new(w.get(name)?, shape),
        device,
    ))
}

fn mm(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    let [m, k] = a.dims();
    let n = b.dims()[1];
    let mut out = None;
    // Match matrixmultiply's 256-wide K blocks, including the intervening
    // FP32 rounding when each block is added to the accumulated output.
    for first in (0..k).step_by(256) {
        let end = (first + 256).min(k);
        let block = Tensor::from_dispatch(<Dispatch as ConvOps>::strict_mm(
            a.clone().slice([0..m, first..end]).into_dispatch(),
            b.clone().slice([first..end, 0..n]).into_dispatch(),
        ));
        out = Some(match out {
            Some(sum) => sum + block,
            None => block,
        });
    }
    out.unwrap()
}

struct Conv {
    w: Tensor<2>,
    b: Tensor<1>,
    kt: usize,
    kf: usize,
    dilation: usize,
    pad: (usize, usize),
    stride: (usize, usize),
}

impl Conv {
    fn load(
        w: &Weights,
        prefix: &str,
        device: &Device,
        dilation: usize,
        pad: (usize, usize),
        stride: (usize, usize),
        along_time: bool,
    ) -> Result<Self> {
        let weight = tensor::<4>(w, &format!("{prefix}.weight"), device)?;
        let [cout, cin, mut kt, mut kf] = weight.dims();
        if along_time {
            std::mem::swap(&mut kt, &mut kf);
        }
        Ok(Self {
            w: weight.reshape([cout, cin * kt * kf]),
            b: tensor(w, &format!("{prefix}.bias"), device)?,
            kt,
            kf,
            dilation,
            pad,
            stride,
        })
    }

    fn forward(&self, x: Tensor<2>, time: usize, freq: usize) -> (Tensor<2>, usize, usize) {
        let to = (time + 2 * self.pad.0 - self.dilation * (self.kt - 1) - 1) / self.stride.0 + 1;
        let fo = (freq + 2 * self.pad.1 - self.kf) / self.stride.1 + 1;
        let positions = to * fo;
        if self.kt == 1 && self.kf == 1 && self.pad == (0, 0) && self.stride == (1, 1) {
            return (
                mm(self.w.clone(), x) + self.b.clone().unsqueeze_dim::<2>(1),
                to,
                fo,
            );
        }
        let mut tiles = Vec::new();
        for first in (0..positions).step_by(16384) {
            let count = (positions - first).min(16384);
            let cols = Tensor::from_dispatch(<Dispatch as ConvOps>::columns(
                x.clone().into_dispatch(),
                time,
                freq,
                self.kt,
                self.kf,
                self.dilation,
                self.pad.0,
                self.pad.1,
                self.stride.0,
                self.stride.1,
                fo,
                first,
                count,
            ));
            tiles.push(mm(self.w.clone(), cols));
        }
        (
            Tensor::cat(tiles, 1) + self.b.clone().unsqueeze_dim::<2>(1),
            to,
            fo,
        )
    }
}

struct Act {
    w: Tensor<1>,
    b: Tensor<1>,
    p: Tensor<1>,
}
impl Act {
    fn load(w: &Weights, prefix: &str, device: &Device) -> Result<Self> {
        Ok(Self {
            w: tensor(w, &format!("{prefix}.1.weight"), device)?,
            b: tensor(w, &format!("{prefix}.1.bias"), device)?,
            p: tensor(w, &format!("{prefix}.2.weight"), device)?,
        })
    }
    fn apply(&self, x: Tensor<2>) -> Tensor<2> {
        Tensor::from_dispatch(<Dispatch as ConvOps>::norm_act(
            x.into_dispatch(),
            self.w.clone().into_dispatch(),
            self.b.clone().into_dispatch(),
            self.p.clone().into_dispatch(),
        ))
    }
}

struct ConvBlock {
    conv: Conv,
    act: Act,
}
impl ConvBlock {
    fn load(w: &Weights, p: &str, device: &Device, conv: Conv) -> Result<Self> {
        Ok(Self {
            conv,
            act: Act::load(w, p, device)?,
        })
    }
    fn forward(&self, x: Tensor<2>, time: usize, freq: usize) -> (Tensor<2>, usize, usize) {
        let (x, t, f) = self.conv.forward(x, time, freq);
        (self.act.apply(x), t, f)
    }
}

struct Dense {
    layers: Vec<ConvBlock>,
}
impl Dense {
    fn load(w: &Weights, p: &str, device: &Device) -> Result<Self> {
        let layers = (0..4)
            .map(|i| {
                let p = format!("{p}.dense_block.{i}");
                let dilation = 1 << i;
                let conv = Conv::load(
                    w,
                    &format!("{p}.0"),
                    device,
                    dilation,
                    (dilation, 1),
                    (1, 1),
                    false,
                )?;
                ConvBlock::load(w, &p, device, conv)
            })
            .collect::<Result<_>>()?;
        Ok(Self { layers })
    }
    fn forward(&self, x: Tensor<2>, time: usize, freq: usize) -> Tensor<2> {
        let mut features = vec![x];
        for (i, layer) in self.layers.iter().enumerate() {
            let x = if features.len() == 1 {
                features[0].clone()
            } else {
                Tensor::cat(features.clone(), 0)
            };
            let (y, _, _) = layer.forward(x, time, freq);
            if i + 1 == self.layers.len() {
                return y;
            }
            features.insert(0, y);
        }
        unreachable!()
    }
}

fn shuffle(x: Tensor<2>, channels: usize, time: usize, freq: usize, along_time: bool) -> Tensor<2> {
    let factor = x.dims()[0] / channels;
    let x = x.reshape([factor, channels, time, freq]);
    let x = if along_time {
        x.permute([1, 2, 0, 3])
    } else {
        x.permute([1, 2, 3, 0])
    };
    x.reshape([channels, time * freq * factor])
}

struct Decoder {
    dense: Dense,
    up1: ConvBlock,
    up2: ConvBlock,
    heads: Vec<(Tensor<2>, Tensor<1>)>,
}
impl Decoder {
    fn load(w: &Weights, p: &str, names: &[&str], device: &Device) -> Result<Self> {
        let up1 = Conv::load(
            w,
            &format!("{p}.up_conv1.0.conv"),
            device,
            1,
            (0, 1),
            (1, 1),
            false,
        )?;
        let up2 = Conv::load(
            w,
            &format!("{p}.up_conv2.0.conv"),
            device,
            1,
            (1, 0),
            (1, 1),
            true,
        )?;
        Ok(Self {
            dense: Dense::load(w, &format!("{p}.dense_block"), device)?,
            up1: ConvBlock::load(w, &format!("{p}.up_conv1"), device, up1)?,
            up2: ConvBlock::load(w, &format!("{p}.up_conv2"), device, up2)?,
            heads: names
                .iter()
                .map(|name| {
                    let weight = tensor::<4>(w, &format!("{p}.{name}.weight"), device)?;
                    let channels = weight.dims()[1];
                    Ok((
                        weight.reshape([1, channels]),
                        tensor(w, &format!("{p}.{name}.bias"), device)?,
                    ))
                })
                .collect::<Result<_>>()?,
        })
    }
    fn forward(&self, x: Tensor<2>, time: usize, freq: usize) -> Vec<Vec<f32>> {
        let channels = x.dims()[0];
        let x = self.dense.forward(x, time, freq);
        let x = self.up1.conv.forward(x, time, freq).0;
        let x = self.up1.act.apply(shuffle(x, channels, time, freq, false));
        let freq = freq * 2;
        let x = self.up2.conv.forward(x, time, freq).0;
        let x = self.up2.act.apply(shuffle(x, channels, time, freq, true));
        self.heads
            .iter()
            .map(|(w, b)| {
                (mm(w.clone(), x.clone()) + b.clone().unsqueeze::<2>())
                    .into_data()
                    .try_into_vec::<f32>()
                    .expect("FP32 decoder output")
            })
            .collect()
    }
}

pub(crate) struct Convolutions {
    device: Device,
    input: ConvBlock,
    dense: Dense,
    output: ConvBlock,
    mag: Decoder,
    pha: Decoder,
}
impl Convolutions {
    pub(crate) fn load(w: &Weights, device: Device) -> Result<Self> {
        let p = "dense_encoder";
        let input = Conv::load(
            w,
            &format!("{p}.dense_conv_1.0"),
            &device,
            1,
            (0, 0),
            (1, 1),
            false,
        )?;
        let output = Conv::load(
            w,
            &format!("{p}.dense_conv_2.0"),
            &device,
            1,
            (0, 0),
            (4, 2),
            false,
        )?;
        Ok(Self {
            input: ConvBlock::load(w, &format!("{p}.dense_conv_1"), &device, input)?,
            dense: Dense::load(w, &format!("{p}.dense_block"), &device)?,
            output: ConvBlock::load(w, &format!("{p}.dense_conv_2"), &device, output)?,
            mag: Decoder::load(w, "mask_decoder", &["final_conv"], &device)?,
            pha: Decoder::load(
                w,
                "phase_decoder",
                &["phase_conv_r", "phase_conv_i"],
                &device,
            )?,
            device,
        })
    }
    pub(crate) fn forward(
        &self,
        x: Vec<f32>,
        time: usize,
        freq: usize,
        mamba: &gpu::Model,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, usize) {
        let x = Tensor::from_data(TensorData::new(x, [2, time * freq]), &self.device);
        let (x, time, freq) = self.input.forward(x, time, freq);
        let x = self.dense.forward(x, time, freq);
        let (x, time, freq) = self.output.forward(x, time, freq);
        let channels = x.dims()[0];
        let rows = x
            .reshape([channels, time, freq])
            .permute([2, 1, 0])
            .reshape([time * freq, channels]);
        let rows = mamba.mamba_tensor(rows, time, freq, channels);
        let x = rows
            .reshape([freq, time, channels])
            .permute([2, 1, 0])
            .reshape([channels, time * freq]);
        let mag = self.mag.forward(x.clone(), time, freq);
        let pha = self.pha.forward(x, time, freq);
        (mag, pha, 2 * freq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_fp32_matmul_and_fp64_norm() {
        if std::env::var("REUSE_GPU").as_deref() != Ok("cuda") {
            return;
        }
        let device = crate::GpuDevice::Cuda(0).device().unwrap();
        let (m, k, n) = (4, 576, 1033);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 % 41) as f32 - 20.0) / 128.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 17 % 127) as f32 - 63.0) / 127.0)
            .collect();
        let mut expected = vec![0.0; m * n];
        crate::ops::gemm(false, false, m, n, k, &a, k, &b, n, 0.0, &mut expected, n);
        let x = mm(
            Tensor::from_data(TensorData::new(a, [m, k]), &device),
            Tensor::from_data(TensorData::new(b, [k, n]), &device),
        );
        let actual = x.to_data().try_into_vec::<f32>().unwrap();
        let max = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(max, 0.0, "match CPU K-block accumulation order");
        let w: Vec<f32> = vec![1.1; m];
        let bias: Vec<f32> = vec![0.1; m];
        let p: Vec<f32> = vec![0.25; m];
        let make = |v: Vec<f32>| Tensor::from_data(TensorData::new(v, [m]), &device);
        let norm = Act {
            w: make(w),
            b: make(bias),
            p: make(p),
        }
        .apply(x)
        .into_data()
        .try_into_vec::<f32>()
        .unwrap();
        let mut host = actual;
        for row in host.chunks_exact_mut(n) {
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            let variance = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
            let scale = 1.1f32 as f64 / (variance + 1e-5).sqrt();
            let shift = 0.1f32 as f64 - mean * scale;
            for v in row {
                let y = *v * scale as f32 + shift as f32;
                *v = if y > 0.0 { y } else { y * 0.25 };
            }
        }
        let max = host
            .iter()
            .zip(&norm)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(max, 0.0, "round multiplication before addition");
    }
}
