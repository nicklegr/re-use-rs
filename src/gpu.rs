//! The offline model's TF-Mamba blocks on a GPU, with Burn. Each Mamba runs as three matmuls
//! and two CubeCL kernels: a causal depthwise conv, and the selective scan with its dt projection
//! and output gate, where one thread per (sequence, channel) walks the whole sequence.
//!
//! The real-time model stays on the CPU: one 20 ms frame per step gives the GPU little work per
//! launch, and waiting for every frame's result makes frame times uneven.

use crate::weights::Weights;
use burn::backend::{backend_extension, tensor::FloatTensor, Backend, Dispatch};
#[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
use burn::tensor::DeviceKind;
use burn::tensor::{Device, Tensor, TensorData};
use burn_cubecl::{
    kernel::into_contiguous,
    ops::numeric::{empty_device_contiguous_dtype, empty_device_dtype},
    tensor::CubeTensor,
    CubeBackend,
};
use cubecl::{calculate_cube_count_elemwise, prelude::CubeDim};
use eyre::Result;
use kernel::{conv_kernel, scan_kernel};

/// Which GPU API to run on. Each one needs its cargo feature (`wgpu`, `metal`, `vulkan`, `cuda`,
/// `rocm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuDevice {
    /// wgpu on whatever the system offers: Metal, Vulkan or DX12.
    Wgpu,
    /// Metal with native MSL shaders (Apple).
    Metal,
    /// Vulkan with SPIR-V shaders.
    Vulkan,
    /// CUDA device with this index (NVIDIA).
    Cuda(usize),
    /// ROCm device with this index (AMD).
    Rocm(usize),
}

impl GpuDevice {
    pub(crate) fn device(self) -> Result<Device> {
        let missing =
            |feature: &str| eyre::eyre!("reuse-rs was built without the `{feature}` feature");
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu => Ok(Device::wgpu(DeviceKind::DefaultDevice)),
            #[cfg(feature = "metal")]
            Self::Metal => Ok(Device::metal(DeviceKind::DefaultDevice)),
            #[cfg(feature = "vulkan")]
            Self::Vulkan => Ok(Device::vulkan(DeviceKind::DefaultDevice)),
            #[cfg(feature = "cuda")]
            Self::Cuda(i) => Ok(Device::cuda(i)),
            #[cfg(feature = "rocm")]
            Self::Rocm(i) => Ok(Device::rocm(i)),
            #[allow(unreachable_patterns)]
            other => Err(missing(match other {
                Self::Wgpu => "wgpu",
                Self::Metal => "metal",
                Self::Vulkan => "vulkan",
                Self::Cuda(_) => "cuda",
                Self::Rocm(_) => "rocm",
            })),
        }
    }
}

#[backend_extension(Cube)]
pub trait MambaOps: Backend {
    fn project(a: FloatTensor<Self>, b: FloatTensor<Self>) -> FloatTensor<Self>;
    #[allow(clippy::too_many_arguments)]
    fn finish_block(
        fw: FloatTensor<Self>,
        bw: FloatTensor<Self>,
        projection_bias: FloatTensor<Self>,
        norm_w: FloatTensor<Self>,
        norm_b: FloatTensor<Self>,
        residual: FloatTensor<Self>,
        batch: usize,
    ) -> FloatTensor<Self>;
    /// Causal depthwise conv over each sequence, then silu. `xz` is `[batch·len, 2·d_inner]`
    /// (the conv reads its first half), `weight` is `[d_inner, taps]`. Returns
    /// `[batch·len, d_inner]`.
    fn mamba_conv(
        xz: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        batch: usize,
        reverse: bool,
    ) -> FloatTensor<Self>;

    /// Selective scan from zero state, with the dt projection, softplus and output gate fused:
    /// returns `(C·s + D·x)·silu(z)` as `[batch·len, d_inner]`. `dbl` rows are
    /// `[dt_raw | B | C]`, `dt_w` is `[d_inner, dt_rank]`, `a` is `[d_inner, 16]`.
    #[allow(clippy::too_many_arguments)]
    fn mamba_scan(
        xz: FloatTensor<Self>,
        xc: FloatTensor<Self>,
        dbl: FloatTensor<Self>,
        dt_w: FloatTensor<Self>,
        dt_b: FloatTensor<Self>,
        a: FloatTensor<Self>,
        d: FloatTensor<Self>,
        batch: usize,
        reverse: bool,
    ) -> FloatTensor<Self>;
}

mod kernel {
    use cubecl::prelude::*;

    /// Match Burn's unit reduction: four lanes, accumulated in channel groups
    /// of four, then reduced left-to-right from zero. Round squares separately.
    #[cube(launch)]
    pub fn finish_kernel<F: Float>(
        fw: &Tensor<F>,
        bw: &Tensor<F>,
        projection_bias: &Tensor<F>,
        norm_w: &Tensor<F>,
        norm_b: &Tensor<F>,
        residual: &Tensor<F>,
        out: &mut Tensor<F>,
        rows: u32,
        batch: u32,
        #[comptime] channels: u32,
        #[define(F)] _dtype: ElemType,
    ) {
        let row = ABSOLUTE_POS;
        if row >= rows as usize {
            terminate!();
        }
        let channels = channels as usize;
        let mut values = Array::<F>::new(channels);
        let mut sums = Array::<F>::new(4usize);
        #[unroll]
        for lane in 0..4usize {
            sums[lane] = F::new(0.0f32);
        }
        #[unroll]
        for group in 0..channels / 4usize {
            #[unroll]
            for lane in 0..4usize {
                let c = group * 4usize + lane;
                let at = row * channels + c;
                let v = (fw[at] + bw[at]) + projection_bias[c];
                values[c] = v;
                sums[lane] += v;
            }
        }
        let mean =
            ((((F::new(0.0f32) + sums[0]) + sums[1]) + sums[2]) + sums[3]) / F::cast_from(channels);
        #[unroll]
        for lane in 0..4usize {
            sums[lane] = F::new(0.0f32);
        }
        #[unroll]
        for group in 0..channels / 4usize {
            #[unroll]
            for lane in 0..4usize {
                let c = group * 4usize + lane;
                let v = values[c] - mean;
                values[c] = v;
                let square = fma(v, v, F::new(0.0f32));
                sums[lane] += square;
            }
        }
        let variance =
            ((((F::new(0.0f32) + sums[0]) + sums[1]) + sums[2]) + sums[3]) / F::cast_from(channels);
        let std = (variance + F::new(1e-5f32)).sqrt();
        let len = rows as usize / batch as usize;
        let dst = (row % len) * batch as usize + row / len;
        #[unroll]
        for c in 0..channels {
            let normalized = values[c] / std;
            let product = fma(normalized, norm_w[c], F::new(0.0f32));
            let v = product + norm_b[c];
            out[dst * channels + c] = residual[row * channels + c] + v;
        }
    }

    /// One thread per (row, channel).
    #[cube(launch)]
    pub fn conv_kernel<F: Float>(
        xz: &Tensor<F>,
        weight: &Tensor<F>,
        bias: &Tensor<F>,
        xc: &mut Tensor<F>,
        rows: u32,
        #[comptime] len: u32,
        #[comptime] d_inner: u32,
        #[comptime] taps: u32,
        #[comptime] reverse: u32,
        #[define(F)] _dtype: ElemType,
    ) {
        let di = d_inner as usize;
        if ABSOLUTE_POS >= rows as usize * di {
            terminate!();
        }
        let row = ABSOLUTE_POS / di;
        let c = ABSOLUTE_POS % di;
        let taps = taps as usize;
        let len = len as usize;
        let l = row % len;
        let mut acc = bias[c];
        #[unroll]
        for k in 0..taps {
            // Tap k reads `taps - 1 - k` steps back; steps before the sequence start are zero.
            let back = taps - 1 - k;
            if reverse == 0 {
                if l >= back {
                    acc += weight[c * taps + k] * xz[(row - back) * 2 * di + c];
                }
            } else if l + back < len {
                acc += weight[c * taps + k] * xz[(row + back) * 2 * di + c];
            }
        }
        xc[row * di + c] = acc / (F::new(1.0f32) + (-acc).exp());
    }

    /// One thread per (sequence, channel), walking the whole sequence.
    #[cube(launch)]
    pub fn scan_kernel<F: Float>(
        xz: &Tensor<F>,
        xc: &Tensor<F>,
        dbl: &Tensor<F>,
        dt_w: &Tensor<F>,
        dt_b: &Tensor<F>,
        a: &Tensor<F>,
        d: &Tensor<F>,
        y: &mut Tensor<F>,
        batch: u32,
        #[comptime] len: u32,
        #[comptime] d_inner: u32,
        #[comptime] dt_rank: u32,
        #[comptime] reverse: u32,
        #[comptime] shared_coefficients: bool,
        #[define(F)] _dtype: ElemType,
    ) {
        let di = d_inner as usize;
        if ABSOLUTE_POS >= batch as usize * di {
            terminate!();
        }
        let seq = ABSOLUTE_POS / di;
        let c = ABSOLUTE_POS % di;
        let (len, rank) = (len as usize, dt_rank as usize);
        let nd = rank + 32;
        let mut s = Array::new(16usize);
        let mut ac = Array::new(16usize);
        let mut dt = Array::new(rank);
        // Each thread owns one column: no barriers are needed. Keeping A in
        // shared memory frees registers without changing the scan arithmetic.
        let mut shared_a = Shared::<[F]>::new_slice(16usize * 32usize);
        #[unroll]
        for k in 0..rank {
            dt[k] = dt_w[c * rank + k];
        }
        #[unroll]
        for n in 0..16usize {
            s[n] = F::new(0.0f32);
            if shared_coefficients {
                shared_a[n * 32usize + UNIT_POS as usize] = a[c * 16 + n];
            } else {
                ac[n] = a[c * 16 + n];
            }
        }
        let (bias, skip) = (dt_b[c], d[c]);
        for i in 0..len {
            let mut l = i;
            if reverse != 0 {
                l = len - 1 - i;
            }
            let row = seq * len + l;
            let mut raw = bias;
            #[unroll]
            for k in 0..rank {
                raw += dbl[row * nd + k] * dt[k];
            }
            // torch softplus, threshold 20.
            let mut delta = raw;
            if raw <= F::new(20.0f32) {
                delta = raw.exp().log1p();
            }
            let x = xc[row * di + c];
            let dx = delta * x;
            let mut acc = F::new(0.0f32);
            #[unroll]
            for n in 0..16usize {
                let coeff = if shared_coefficients {
                    shared_a[n * 32usize + UNIT_POS as usize]
                } else {
                    ac[n]
                };
                let decay = (delta * coeff).exp();
                s[n] = decay * s[n] + dx * dbl[row * nd + rank + n];
                acc += s[n] * dbl[row * nd + rank + 16 + n];
            }
            let z = xz[row * 2 * di + di + c];
            y[row * di + c] = (acc + skip * x) * (z / (F::new(1.0f32) + (-z).exp()));
        }
    }
}

const BLOCK: u32 = 256;
// A scan keeps 16 state values live per thread; one warp per block allows
// more independent blocks to fit when registers limit occupancy.
const SCAN_BLOCK: u32 = 32;

impl MambaOps for CubeBackend {
    fn project(a: CubeTensor, b: CubeTensor) -> CubeTensor {
        if a.client.name() == "cuda" {
            if let Ok(out) = cuda_project(a.clone(), b.clone()) {
                return out;
            }
        }
        let dtype = a.dtype;
        burn_cubecl::kernel::matmul::matmul(a, b, None, Default::default(), dtype)
            .expect("Mamba matmul")
    }
    fn finish_block(
        fw: CubeTensor,
        bw: CubeTensor,
        projection_bias: CubeTensor,
        norm_w: CubeTensor,
        norm_b: CubeTensor,
        residual: CubeTensor,
        batch: usize,
    ) -> CubeTensor {
        let [fw, bw, projection_bias, norm_w, norm_b, residual] =
            [fw, bw, projection_bias, norm_w, norm_b, residual].map(into_contiguous);
        let (rows, channels) = (fw.meta.shape()[0], fw.meta.shape()[1]);
        assert!(channels % 4 == 0 && batch > 0 && rows % batch == 0);
        let out = empty_device_contiguous_dtype(
            fw.client.clone(),
            fw.device.clone(),
            [rows, channels].into(),
            fw.dtype,
        );
        let dtype = burn::backend::cubecl::dtype_to_storage_type(fw.dtype);
        kernel::finish_kernel::launch(
            &fw.client,
            calculate_cube_count_elemwise(&fw.client, rows, CubeDim::new_1d(64)),
            CubeDim::new_1d(64),
            fw.clone().into_tensor_arg(),
            bw.into_tensor_arg(),
            projection_bias.into_tensor_arg(),
            norm_w.into_tensor_arg(),
            norm_b.into_tensor_arg(),
            residual.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            rows as u32,
            batch as u32,
            channels as u32,
            dtype,
        );
        out
    }
    fn mamba_conv(
        xz: CubeTensor,
        weight: CubeTensor,
        bias: CubeTensor,
        batch: usize,
        reverse: bool,
    ) -> CubeTensor {
        let (xz, weight, bias) = (
            into_contiguous(xz),
            into_contiguous(weight),
            into_contiguous(bias),
        );
        let rows = xz.meta.shape()[0];
        let (di, taps) = (weight.meta.shape()[0], weight.meta.shape()[1]);
        let xc = empty_device_dtype(
            xz.client.clone(),
            xz.device.clone(),
            [rows, di].into(),
            xz.dtype,
        );
        let threads = rows * di;
        let dtype = burn::backend::cubecl::dtype_to_storage_type(xz.dtype);
        let client = xz.client.clone();
        conv_kernel::launch(
            &client,
            calculate_cube_count_elemwise(&client, threads, CubeDim::new_1d(BLOCK)),
            CubeDim::new_1d(BLOCK),
            xz.into_tensor_arg(),
            weight.into_tensor_arg(),
            bias.into_tensor_arg(),
            xc.clone().into_tensor_arg(),
            rows as u32,
            (rows / batch) as u32,
            di as u32,
            taps as u32,
            reverse as u32,
            dtype,
        );
        xc
    }

    fn mamba_scan(
        xz: CubeTensor,
        xc: CubeTensor,
        dbl: CubeTensor,
        dt_w: CubeTensor,
        dt_b: CubeTensor,
        a: CubeTensor,
        d: CubeTensor,
        batch: usize,
        reverse: bool,
    ) -> CubeTensor {
        let [xz, xc, dbl, dt_w, dt_b, a, d] = [xz, xc, dbl, dt_w, dt_b, a, d].map(into_contiguous);
        let (rows, di) = (xc.meta.shape()[0], xc.meta.shape()[1]);
        let rank = dt_w.meta.shape()[1];
        let y = empty_device_dtype(
            xc.client.clone(),
            xc.device.clone(),
            [rows, di].into(),
            xc.dtype,
        );
        let threads = batch * di;
        let dtype = burn::backend::cubecl::dtype_to_storage_type(xc.dtype);
        let client = xc.client.clone();
        scan_kernel::launch(
            &client,
            calculate_cube_count_elemwise(&client, threads, CubeDim::new_1d(SCAN_BLOCK)),
            CubeDim::new_1d(SCAN_BLOCK),
            xz.into_tensor_arg(),
            xc.into_tensor_arg(),
            dbl.into_tensor_arg(),
            dt_w.into_tensor_arg(),
            dt_b.into_tensor_arg(),
            a.into_tensor_arg(),
            d.into_tensor_arg(),
            y.clone().into_tensor_arg(),
            batch as u32,
            (rows / batch) as u32,
            di as u32,
            rank as u32,
            reverse as u32,
            client.name() == "cuda",
            dtype,
        );
        y
    }
}

fn tensor<const D: usize>(w: &Weights, name: &str, device: &Device) -> Result<Tensor<D>> {
    let shape: [usize; D] = w
        .shape(name)?
        .try_into()
        .map_err(|_| eyre::eyre!("{name}: expected {D} dimensions"))?;
    Ok(Tensor::from_data(
        TensorData::new(w.get(name)?, shape),
        device,
    ))
}

fn project(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as MambaOps>::project(
        a.into_dispatch(),
        b.into_dispatch(),
    ))
}

// Keep Burn's TF32 CMMA instruction and K iteration order, but use twice as
// many row partitions and tilewise loading. Launch the prepared blueprint
// directly: CubeK's forced-strategy path otherwise resets its TF32 types.
fn cuda_project(
    a: CubeTensor,
    b: CubeTensor,
) -> Result<CubeTensor, cubek::matmul::definition::MatmulSetupError> {
    use cubek::matmul::{
        definition::{AvailableVectorSizes, MatmulElems, MatmulGlobalElems, MatmulProblem},
        multi_level::{
            args::{ConcreteInputsFactory, ConcreteOutputFactory, InputArg, OutputArg, TensorArgs},
            components::{
                global::read::sync_full_tilewise::SyncFullTilewiseLoading, tile::TileMatmulKind,
            },
            launch_kernel,
            routines::{
                batch::simple::SimpleAlgorithm, infer_blueprint_plane, PlaneTilingBlueprintOptions,
            },
            tile::{ColMajorTilingOrder, RowMajorTilingOrder},
            BatchMatmulRoutine, ExpandInfo,
        },
    };
    type Algo = SimpleAlgorithm<
        SyncFullTilewiseLoading<ColMajorTilingOrder>,
        SyncFullTilewiseLoading<RowMajorTilingOrder>,
    >;
    let out = empty_device_dtype(
        a.client.clone(),
        a.device.clone(),
        [a.meta.shape()[0], b.meta.shape()[1]].into(),
        a.dtype,
    );
    let dtype = burn::backend::cubecl::dtype_to_storage_type(a.dtype);
    let globals = MatmulGlobalElems {
        lhs: dtype,
        rhs: dtype,
        out: dtype,
    };
    let problem = MatmulProblem::from_shapes_and_strides(
        a.meta.shape().clone(),
        b.meta.shape().clone(),
        out.meta.shape().clone(),
        a.meta.strides().clone(),
        b.meta.strides().clone(),
        out.meta.strides().clone(),
        globals.clone(),
        cubecl::ir::AddressType::U32,
        None,
        None,
    )?;
    let vectors = AvailableVectorSizes::from_type_sizes(&out.client, 4, 4, 4)
        .filter_lhs_with_tensor(&problem.lhs_strides, &problem.lhs_shape, problem.lhs_layout)
        .filter_rhs_with_tensor(&problem.rhs_strides, &problem.rhs_shape, problem.rhs_layout)
        .filter_out_with_tensor(&problem.out_strides, &problem.out_shape)
        .pick_max()?;
    let (mut blueprint, dtypes) = infer_blueprint_plane(
        TileMatmulKind::Cmma,
        &out.client,
        &problem,
        32,
        MatmulElems::from_globals(&globals),
        &vectors,
        PlaneTilingBlueprintOptions::default(),
    )?;
    blueprint.tiling_scheme.stage_size.m *= 2;
    blueprint.check_m_bounds = problem.m
        % blueprint
            .tiling_scheme
            .elements_per_global_partition_along_m() as usize
        != 0;
    let settings = <Algo as BatchMatmulRoutine<()>>::device_settings(&out.client, vectors);
    let info = <Algo as BatchMatmulRoutine<()>>::prepare(
        &problem,
        &settings,
        ExpandInfo {
            blueprint,
            dtypes: dtypes.clone(),
        },
    )?;
    let input = <InputArg<TensorArgs> as ConcreteInputsFactory<Algo>>::create(
        cubek::std::InputBinding::new(a.binding(), dtype),
        cubek::std::InputBinding::new(b.binding(), dtype),
        &info.blueprint,
        &problem,
        &info.vector_sizes,
        &dtypes,
    );
    let output = <OutputArg<TensorArgs> as ConcreteOutputFactory<Algo>>::create(
        out.clone().binding(),
        &info.blueprint,
        &problem,
        &info.vector_sizes,
        &dtypes,
    );
    launch_kernel::<TensorArgs, Algo>(&out.client, input, output, (), info)?;
    Ok(out)
}

/// `[out, in]` weight as `[in, out]`, for `x.matmul(w)`.
fn linear(w: &Weights, name: &str, device: &Device) -> Result<Tensor<2>> {
    Ok(tensor::<2>(w, name, device)?.transpose())
}

struct Mamba {
    in_proj: Tensor<2>,
    conv_w: Tensor<2>,
    conv_b: Tensor<1>,
    x_proj: Tensor<2>,
    dt_w: Tensor<2>,
    dt_b: Tensor<1>,
    a: Tensor<2>,
    d: Tensor<1>,
    out_proj: Tensor<2>,
}

impl Mamba {
    fn load(w: &Weights, prefix: &str, device: &Device) -> Result<Self> {
        let conv_w = tensor::<3>(w, &format!("{prefix}.conv1d.weight"), device)?;
        let [di, _, taps] = conv_w.dims();
        Ok(Self {
            in_proj: linear(w, &format!("{prefix}.in_proj.weight"), device)?,
            conv_w: conv_w.reshape([di, taps]),
            conv_b: tensor(w, &format!("{prefix}.conv1d.bias"), device)?,
            x_proj: linear(w, &format!("{prefix}.x_proj.weight"), device)?,
            dt_w: tensor(w, &format!("{prefix}.dt_proj.weight"), device)?,
            dt_b: tensor(w, &format!("{prefix}.dt_proj.bias"), device)?,
            a: tensor::<2>(w, &format!("{prefix}.A_log"), device)?
                .exp()
                .neg(),
            d: tensor(w, &format!("{prefix}.D"), device)?,
            out_proj: linear(w, &format!("{prefix}.out_proj.weight"), device)?,
        })
    }

    /// `x` is `[batch·len, d_model]`, sequence-major.
    fn forward(&self, x: Tensor<2>, batch: usize, reverse: bool) -> Tensor<2> {
        let xz = project(x, self.in_proj.clone());
        let xc = Tensor::<2>::from_dispatch(<Dispatch as MambaOps>::mamba_conv(
            xz.clone().into_dispatch(),
            self.conv_w.clone().into_dispatch(),
            self.conv_b.clone().into_dispatch(),
            batch,
            reverse,
        ));
        let dbl = project(xc.clone(), self.x_proj.clone());
        let y = Tensor::<2>::from_dispatch(<Dispatch as MambaOps>::mamba_scan(
            xz.into_dispatch(),
            xc.into_dispatch(),
            dbl.into_dispatch(),
            self.dt_w.clone().into_dispatch(),
            self.dt_b.clone().into_dispatch(),
            self.a.clone().into_dispatch(),
            self.d.clone().into_dispatch(),
            batch,
            reverse,
        ));
        project(y, self.out_proj.clone())
    }
}

/// LayerNorm over the last dimension (eps 1e-5).
fn layer_norm(x: Tensor<2>, weight: &Tensor<1>, bias: &Tensor<1>) -> Tensor<2> {
    let mean = x.clone().mean_dim(1);
    let centered = x - mean;
    let var = centered.clone().square().mean_dim(1);
    centered / (var + 1e-5).sqrt() * weight.clone().unsqueeze::<2>() + bias.clone().unsqueeze::<2>()
}

/// Bidirectional MambaBlock: `norm(output_proj([fw(x) + x, flip(bw(flip x)) + x]))`. The
/// backward Mamba walks each sequence from its end, and `output_proj` is split into the halves
/// that multiply each direction.
struct BiMamba {
    forward: Mamba,
    backward: Mamba,
    proj_fw: Tensor<2>,
    proj_bw: Tensor<2>,
    output_bias: Tensor<1>,
    norm_w: Tensor<1>,
    norm_b: Tensor<1>,
    streams: Option<[cubecl_environment::stream::StreamId; 2]>,
}

impl BiMamba {
    fn load(w: &Weights, prefix: &str, device: &Device, parallel: bool) -> Result<Self> {
        let proj = linear(w, &format!("{prefix}.output_proj.weight"), device)?;
        let [inp, out] = proj.dims();
        Ok(Self {
            forward: Mamba::load(w, &format!("{prefix}.forward_blocks"), device)?,
            backward: Mamba::load(w, &format!("{prefix}.backward_blocks"), device)?,
            proj_fw: proj.clone().slice([0..inp / 2, 0..out]),
            proj_bw: proj.slice([inp / 2..inp, 0..out]),
            output_bias: tensor(w, &format!("{prefix}.output_proj.bias"), device)?,
            norm_w: tensor(w, &format!("{prefix}.norm.weight"), device)?,
            norm_b: tensor(w, &format!("{prefix}.norm.bias"), device)?,
            streams: if parallel {
                use cubecl_environment::stream::StreamId;
                static STREAMS: std::sync::OnceLock<[StreamId; 2]> = std::sync::OnceLock::new();
                Some(*STREAMS.get_or_init(|| [StreamId::allocate(), StreamId::allocate()]))
            } else {
                None
            },
        })
    }

    fn forward(&self, x: Tensor<2>, batch: usize) -> Tensor<2> {
        if let Some([fw_stream, bw_stream]) = self.streams {
            let fw = fw_stream.executes(|| {
                project(
                    self.forward.forward(x.clone(), batch, false) + x.clone(),
                    self.proj_fw.clone(),
                )
            });
            let bw = bw_stream.executes(|| {
                project(
                    self.backward.forward(x.clone(), batch, true) + x,
                    self.proj_bw.clone(),
                )
            });
            // CubeCL tracks cross-stream dependencies of these tensors. Keep
            // the projection sum and normalization in their original order.
            let out = fw + bw + self.output_bias.clone().unsqueeze::<2>();
            return layer_norm(out, &self.norm_w, &self.norm_b);
        }
        let fw = self.forward.forward(x.clone(), batch, false) + x.clone();
        let bw = self.backward.forward(x.clone(), batch, true) + x;
        let out = fw.matmul(self.proj_fw.clone())
            + bw.matmul(self.proj_bw.clone())
            + self.output_bias.clone().unsqueeze::<2>();
        layer_norm(out, &self.norm_w, &self.norm_b)
    }

    fn forward_transposed(&self, x: Tensor<2>, batch: usize) -> Tensor<2> {
        let [fw_stream, bw_stream] = self.streams.expect("CUDA direction streams");
        let fw = fw_stream.executes(|| {
            project(
                self.forward.forward(x.clone(), batch, false) + x.clone(),
                self.proj_fw.clone(),
            )
        });
        let bw = bw_stream.executes(|| {
            project(
                self.backward.forward(x.clone(), batch, true) + x.clone(),
                self.proj_bw.clone(),
            )
        });
        Tensor::from_dispatch(<Dispatch as MambaOps>::finish_block(
            fw.into_dispatch(),
            bw.into_dispatch(),
            self.output_bias.clone().into_dispatch(),
            self.norm_w.clone().into_dispatch(),
            self.norm_b.clone().into_dispatch(),
            x.into_dispatch(),
            batch,
        ))
    }
}

struct Block {
    time: BiMamba,
    freq: BiMamba,
}

/// The TF-Mamba blocks on a GPU. CUDA also uses the FP32 convolutional path;
/// other backends keep the convolutional encoder and decoders on the CPU.
pub(crate) struct Model {
    blocks: Vec<Block>,
    device: Device,
    fused_finish: bool,
}

impl Model {
    pub(crate) fn load(w: &Weights, blocks: usize, device: GpuDevice) -> Result<Self> {
        let parallel = matches!(device, GpuDevice::Cuda(_));
        let device = device.device()?;
        let blocks = (0..blocks)
            .map(|i| {
                Ok(Block {
                    time: BiMamba::load(w, &format!("TSMamba.{i}.time_mamba"), &device, parallel)?,
                    freq: BiMamba::load(w, &format!("TSMamba.{i}.freq_mamba"), &device, parallel)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            blocks,
            device,
            fused_finish: parallel,
        })
    }

    /// `rows` is `[freq][time][channels]`; returns the same layout.
    pub(crate) fn mamba(
        &self,
        rows: &[f32],
        time: usize,
        freq: usize,
        channels: usize,
    ) -> Vec<f32> {
        let positions = time * freq;
        let x: Tensor<2> = Tensor::from_data(
            TensorData::new(rows.to_vec(), [positions, channels]),
            &self.device,
        );
        self.mamba_tensor(x, time, freq, channels)
            .into_data()
            .try_into_vec::<f32>()
            .expect("GPU output is f32")
    }

    pub(crate) fn mamba_tensor(
        &self,
        mut x: Tensor<2>,
        time: usize,
        freq: usize,
        channels: usize,
    ) -> Tensor<2> {
        if self.fused_finish {
            for block in &self.blocks {
                x = block.time.forward_transposed(x, freq);
                x = block.freq.forward_transposed(x, time);
            }
            return x;
        }
        let positions = time * freq;
        // [a][b][c] -> [b][a][c]
        let swap = |t: Tensor<2>, a: usize, b: usize| {
            t.reshape([a, b, channels])
                .swap_dims(0, 1)
                .reshape([positions, channels])
        };
        for block in &self.blocks {
            x = x.clone() + block.time.forward(x, freq);
            x = swap(x, freq, time);
            x = x.clone() + block.freq.forward(x, time);
            x = swap(x, time, freq);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_project_matches_original_cmma() {
        if std::env::var("REUSE_GPU").as_deref() != Ok("cuda") {
            return;
        }
        let device = GpuDevice::Cuda(0).device().unwrap();
        for (rows, k, n) in [(129, 64, 512), (257, 256, 36), (130, 256, 64)] {
            let a: Vec<f32> = (0..rows * k)
                .map(|i| ((i * 17 % 127) as f32 - 63.0) / 127.0)
                .collect();
            let b: Vec<f32> = (0..n * k)
                .map(|i| ((i * 7 % 41) as f32 - 20.0) / 128.0)
                .collect();
            let a = Tensor::<2>::from_data(TensorData::new(a, [rows, k]), &device);
            let b = Tensor::<2>::from_data(TensorData::new(b, [n, k]), &device).transpose();
            let expected = a
                .clone()
                .matmul(b.clone())
                .into_data()
                .try_into_vec::<f32>()
                .unwrap();
            let actual = Tensor::<2>::from_primitive::<CubeBackend>(
                cuda_project(
                    a.try_into_primitive::<CubeBackend>().unwrap(),
                    b.try_into_primitive::<CubeBackend>().unwrap(),
                )
                .unwrap(),
            )
            .into_data()
            .try_into_vec::<f32>()
            .unwrap();
            assert_eq!(actual, expected);
        }
    }
}
