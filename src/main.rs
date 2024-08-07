use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use core::cell::UnsafeCell;
use galois::error::GError;

use galois::Shape;
use galois::Tensor as GsTensor;
use galois::F16;
use galois::{DType, GS_TYPE_SIZE};
use lazy_static::lazy_static;
use num_traits::float::Float;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Error as IOError;
use std::io::Read;
use std::ptr::NonNull;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::vec;
use thiserror::Error;

const WHISPER_SAMPLE_RATE: usize = 16000;
const WHISPER_N_FFT: usize = 400;
const WHISPER_N_MEL: usize = 80;
const WHISPER_HOP_LENGTH: usize = 160;
const WHISPER_CHUNK_SIZE: usize = 30;

fn get_type_size(t: DType) -> usize {
    return GS_TYPE_SIZE[t as usize];
}

macro_rules! function {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let name = type_name_of(f);
        name.strip_suffix("::f").unwrap()
    }};
}

const MAGIC: u32 = 0x67676d6c;

type Endian = LittleEndian;

#[derive(Error, Debug)]
pub enum WsError {
    #[error("Unexpected: {0}")]
    Unexpected(String),
    #[error("Unexpected IO: {0}")]
    UnexpectIO(io::Error),
    #[error("invalid model file '{0}' (bad magic)\n")]
    BadMagic(String),
    #[error("not enough space in the context's memory pool\n")]
    NotEnoughSpace,
    #[error("unknown tensor '{0}' in model file\n")]
    UnknownTensor(String),
    #[error("invalid ref tensor '{0}'\n")]
    BadRefTensor(String),
    #[error("tensor {0} has wrong size in model file, got:{1}, expected:{2}\n")]
    WrongSizeTensor(String, usize, usize),
    #[error("tensor {0} has wrong shape in model file, got:{1:?}, expected:{2:?}\n")]
    WrongShapeTensor(String, Vec<usize>, Vec<usize>),
    #[error("tensor {0} has wrong bytes in model file, got:{1:?}, expected:{2:?}\n")]
    WrongBytesTensor(String, usize, usize),
    #[error("galois tensor:'{0}'")]
    WrongGTensor(GError),
}

impl From<IOError> for WsError {
    fn from(e: IOError) -> Self {
        WsError::UnexpectIO(e)
    }
}

impl From<&str> for WsError {
    fn from(e: &str) -> Self {
        WsError::Unexpected(e.to_string())
    }
}

impl From<GError> for WsError {
    fn from(e: GError) -> Self {
        WsError::WrongGTensor(e)
    }
}

pub type WsResult<T> = Result<T, WsError>;

#[derive(Hash, Eq, PartialEq, Debug, Default)]
enum EModel {
    #[default]
    Unknown,
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}
impl EModel {
    fn from_audio_layer(n_audio_layer: i32) -> EModel {
        match n_audio_layer {
            4 => EModel::Tiny,
            6 => EModel::Base,
            12 => EModel::Small,
            24 => EModel::Medium,
            32 => EModel::Large,
            _ => EModel::Unknown,
        }
    }
}

const MB: usize = 1024 * 1024;

lazy_static! {
    static ref MEM_REQ_MODEL: HashMap<EModel, usize> = {
        let mut map = HashMap::new();
        map.insert(EModel::Tiny, 74 * MB);
        map.insert(EModel::Base, 142 * MB);
        map.insert(EModel::Small, 466 * MB);
        map.insert(EModel::Medium, 1464 * MB);
        map.insert(EModel::Large, 2952 * MB);
        map
    };
}

lazy_static! {
    static ref MEM_REQ_MEMORY: HashMap<EModel, usize> = {
        let mut map = HashMap::new();
        map.insert(EModel::Tiny, 12 * MB);
        map.insert(EModel::Base, 24 * MB);
        map.insert(EModel::Small, 70 * MB);
        map.insert(EModel::Medium, 184 * MB);
        map.insert(EModel::Large, 306 * MB);
        map
    };
}

lazy_static! {
    static ref MEM_REQ_ENCODE: HashMap<EModel, usize> = {
        let mut map = HashMap::new();
        map.insert(EModel::Tiny, 80 * MB);
        map.insert(EModel::Base, 128 * MB);
        map.insert(EModel::Small, 300 * MB);
        map.insert(EModel::Medium, 680 * MB);
        map.insert(EModel::Large, 1100 * MB);
        map
    };
}

lazy_static! {
    static ref MEM_REQ_ENCODE_LAYER: HashMap<EModel, usize> = {
        let mut map = HashMap::new();
        map.insert(EModel::Tiny, 104 * MB);
        map.insert(EModel::Base, 138 * MB);
        map.insert(EModel::Small, 208 * MB);
        map.insert(EModel::Medium, 280 * MB);
        map.insert(EModel::Large, 354 * MB);
        map
    };
}

lazy_static! {
    static ref MEM_REQ_DECODE: HashMap<EModel, usize> = {
        let mut map = HashMap::new();
        map.insert(EModel::Tiny, 200 * MB);
        map.insert(EModel::Base, 202 * MB);
        map.insert(EModel::Small, 204 * MB);
        map.insert(EModel::Medium, 206 * MB);
        map.insert(EModel::Large, 208 * MB);
        map
    };
}

lazy_static! {
    static ref MEM_REQ_DECODE_LAYER: HashMap<EModel, usize> = {
        let mut map = HashMap::new();
        map.insert(EModel::Tiny, 32 * MB);
        map.insert(EModel::Base, 44 * MB);
        map.insert(EModel::Small, 64 * MB);
        map.insert(EModel::Medium, 84 * MB);
        map.insert(EModel::Large, 110 * MB);
        map
    };
}

fn new_tensor_1d(ctx: &mut TensorContext, dtype: DType, ne0: usize) -> WsResult<GsTensor> {
    let dim = [ne0];
    new_tensor(ctx, 1, dtype, Shape::from_array(dim))
}

fn new_tensor_2d(
    ctx: &mut TensorContext,
    dtype: DType,
    ne0: usize,
    ne1: usize,
) -> WsResult<GsTensor> {
    let dim = [ne0, ne1];
    new_tensor(ctx, 2, dtype, Shape::from_array(dim))
}

fn new_tensor_3d(
    ctx: &mut TensorContext,
    dtype: DType,
    ne0: usize,
    ne1: usize,
    ne2: usize,
) -> WsResult<GsTensor> {
    let dim = [ne0, ne1, ne2];
    new_tensor(ctx, 3, dtype, Shape::from_array(dim))
}

fn new_tensor_4d(
    ctx: &mut TensorContext,
    dtype: DType,
    ne0: usize,
    ne1: usize,
    ne3: usize,
    ne4: usize,
) -> WsResult<GsTensor> {
    let dim = [ne0, ne1];
    new_tensor(ctx, 2, dtype, Shape::from_array(dim))
}

fn new_f32_tensor(ctx: &mut TensorContext, value: f32) -> WsResult<GsTensor> {
    let mut result = new_tensor_1d(ctx, DType::F32, 1)?;
    result.set_value(value);
    Ok(result)
}

fn new_tensor(
    ctx: &mut TensorContext,
    n_dims: usize,
    dtype: DType,
    shape: Shape,
) -> WsResult<GsTensor> {
    let cur_offset = ctx.offset;
    let cur_size = ctx.size;
    let size_needed: usize = get_type_size(dtype) * shape.size();
    if cur_offset + size_needed > ctx.buf.len() {
        return Err(WsError::NotEnoughSpace);
    }
    let t = unsafe {
        GsTensor::from_bytes(
            &ctx.buf[cur_offset..cur_offset + size_needed],
            n_dims,
            shape,
            dtype,
        )
    };
    ctx.offset = cur_offset + size_needed;
    ctx.size = size_needed;
    ctx.n_objects += 1;
    Ok(t)
}

fn view_tensor(buf: &[u8], n_dims: usize, dtype: DType, shape: Shape) -> WsResult<GsTensor> {
    Ok(unsafe { GsTensor::from_bytes(buf, n_dims, shape, dtype) })
}

fn dup_tensor(ctx: &mut TensorContext, a: &GsTensor) -> WsResult<GsTensor> {
    let dtype = a.dtype();
    let shape = Shape::from_slice(a.dim().shape());
    new_tensor(ctx, a.n_dims(), dtype, shape)
}

fn view_1d(a: &GsTensor, ne0: usize, offset: usize) -> WsResult<GsTensor> {
    let dtype = a.dtype();
    let buf = a.as_bytes();
    let shape = Shape::from_array([ne0]);
    view_tensor(&buf[offset..], 1, dtype, shape)
}

fn view_2d(a: &GsTensor, ne0: usize, ne1: usize, nb1: usize, offset: usize) -> WsResult<GsTensor> {
    let dtype = a.dtype();
    let buf = a.as_bytes();
    let shape = Shape::from_array([ne0, ne1]);
    let mut t = view_tensor(&buf[offset..], 2, dtype, shape)?;
    let nb0 = t.dim().stride_1d();
    let nb = [nb0, nb1, nb1 * ne1, nb1 * ne1];
    t.ret_stride(nb);
    Ok(t)
}

fn reshape_3d(a: &GsTensor, ne0: usize, ne1: usize, ne2: usize) -> WsResult<GsTensor> {
    assert!(a.ggml_is_contiguous());
    assert!(a.elem_count() == ne0 * ne1 * ne2);
    let ne: [usize; 3] = [ne0, ne1, ne2];
    let result = view_tensor(a.as_bytes(), a.n_dims(), a.dtype(), Shape::from_array(ne))?;
    Ok(result)
}

struct TensorContext<'a> {
    offset: usize,
    size: usize,
    n_objects: usize,
    buf: &'a [u8],
}

impl<'a> TensorContext<'a> {
    fn new(buf: &'a [u8]) -> TensorContext<'a> {
        TensorContext {
            offset: 0,
            size: 0,
            n_objects: 0,
            buf: buf,
        }
    }
}

type WhisperToken = i32;

struct WhisperTokenData {
    id: WhisperToken,  // token id
    tid: WhisperToken, // forced timestamp token id

    p: f32,     // probability of the token
    pt: f32,    // probability of the timestamp token
    ptsum: f32, // sum of probabilities of all timestamp tokens

    // token-level timestamp data
    // do not use if you haven't computed token-level timestamps
    t0: i64, // start time of the token
    t1: i64, // end time of the token

    vlen: f32, // voice length of the token
}

struct WhisperContext {
    t_load_us: i64,
    t_mel_us: i64,
    t_sample_us: i64,
    t_encode_us: i64,
    t_decode_us: i64,
    t_start_us: i64,

    buf_model: Vec<u8>, // the model buffer is read-only and can be shared between processors
    buf_memory: Vec<u8>,
    buf_compute: Vec<u8>,
    buf_compute_layer: Vec<u8>,

    model: WhisperModel,
    vocab: WhisperVocab,

    mel: WhisperMel,

    probs: Vec<f32>,
    logits: Vec<f32>,

    result_all: Vec<WhisperSegment>,

    prompt_past: Vec<WhisperToken>,

    t_beg: i64,
    t_last: i64,
    tid_last: WhisperToken,
    energy: Vec<f32>,
    exp_n_audio_ctx: i32,
}

impl WhisperContext {
    fn new(fname: &str) -> WsResult<WhisperContext> {
        let mut fin = open_file_stream(fname)?;
        let magic = fin.read_u32::<Endian>()?;
        if magic != MAGIC {
            return Err(WsError::BadMagic(fname.to_string()));
        }
        let hparams = WhisperHparams::load(&mut fin)?;
        let mtype = EModel::from_audio_layer(hparams.n_audio_layer);
        println!("{}: type           = {:?}\n", function!(), mtype);
        println!(
            "{}: buf_model_size           = {:?}MB\n",
            function!(),
            *MEM_REQ_MODEL.get(&mtype).unwrap() / MB
        );
        println!(
            "{}: buf_memory_size          = {:?}MB\n",
            function!(),
            *MEM_REQ_MEMORY.get(&mtype).unwrap() / MB
        );
        println!(
            "{}: buf_compute_size         = {:?}MB\n",
            function!(),
            std::cmp::max(
                *MEM_REQ_ENCODE.get(&mtype).unwrap(),
                *MEM_REQ_DECODE.get(&mtype).unwrap()
            ) / MB
        );
        println!(
            "{}: buf_compute_layer_size   = {:?}MB\n",
            function!(),
            std::cmp::max(
                *MEM_REQ_ENCODE_LAYER.get(&mtype).unwrap(),
                *MEM_REQ_DECODE_LAYER.get(&mtype).unwrap()
            ) / MB
        );

        let buf_model = vec![0u8; *MEM_REQ_MODEL.get(&mtype).unwrap()];
        let buf_memory = vec![0u8; *MEM_REQ_MEMORY.get(&mtype).unwrap()];

        let buf_compute = vec![
            0u8;
            std::cmp::max(
                *MEM_REQ_ENCODE.get(&mtype).unwrap(),
                *MEM_REQ_DECODE.get(&mtype).unwrap(),
            )
        ];
        let buf_compute_layer = vec![
            0u8;
            std::cmp::max(
                *MEM_REQ_ENCODE_LAYER.get(&mtype).unwrap(),
                *MEM_REQ_DECODE_LAYER.get(&mtype).unwrap(),
            )
        ];

        {
            let mem_required =
                buf_model.len() + buf_memory.len() + buf_compute.len() + buf_compute_layer.len();
            println!(
                "{}: mem_required  = {:7.2} MB",
                function!(),
                mem_required as f64 / 1024.0 / 1024.0
            );
        }
        let filters = WhisperFilters::load(&mut fin)?;
        let n_vocab = fin.read_i32::<Endian>()?;
        let mut vocab = WhisperVocab::default().load(n_vocab, &mut fin)?;
        vocab.n_vocab = hparams.n_vocab;
        if vocab.is_multilingual() {
            vocab.token_eot += 1;
            vocab.token_sot += 1;
            vocab.token_prev += 1;
            vocab.token_solm += 1;
            vocab.token_not += 1;
            vocab.token_beg += 1;
        }

        if n_vocab < hparams.n_vocab {
            println!(
                "{}: adding {} extra tokens",
                function!(),
                hparams.n_vocab - n_vocab
            );
            for i in n_vocab..hparams.n_vocab {
                let word = if i > vocab.token_beg {
                    format!("[_TT_{}]", i - vocab.token_beg)
                } else if i == vocab.token_eot {
                    "[_EOT_]".to_string()
                } else if i == vocab.token_sot {
                    "[_SOT_]".to_string()
                } else if i == vocab.token_prev {
                    "[_PREV_]".to_string()
                } else if i == vocab.token_not {
                    "[_NOT_]".to_string()
                } else if i == vocab.token_beg {
                    "[_BEG_]".to_string()
                } else {
                    format!("[_extra_token_{}]", i)
                };
                vocab.token_to_id.insert(word.clone(), i);
                vocab.id_to_token.insert(i, word);
            }
        }
        println!("{}: n_vocab       = {}\n", function!(), n_vocab);
        let whisper_model =
            WhisperModel::load(&mut fin, mtype, hparams, filters, &buf_model, &buf_memory)?;
        Ok(WhisperContext {
            t_load_us: 0,
            t_mel_us: 0,
            t_sample_us: 0,
            t_encode_us: 0,
            t_decode_us: 0,
            t_start_us: 0,

            buf_model: buf_model, // the model buffer is read-only and can be shared between processors
            buf_memory: buf_memory,
            buf_compute: buf_compute,
            buf_compute_layer: buf_compute_layer,

            model: whisper_model,
            vocab: vocab,

            mel: WhisperMel::new(),

            probs: Vec::new(),
            logits: Vec::new(),

            result_all: Vec::new(),

            prompt_past: Vec::new(),

            t_beg: 0,
            t_last: 0,
            tid_last: 0,
            energy: Vec::new(),
            exp_n_audio_ctx: 0,
        })
    }
}

#[derive(Default)]
struct WhisperFilters {
    n_mel: i32,
    n_ff: i32,
    data: Arc<MthVecF32>,
}

impl WhisperFilters {
    fn load<T: Read>(r: &mut T) -> WsResult<WhisperFilters> {
        let n_mel: i32 = r.read_i32::<Endian>()?;
        let n_ff: i32 = r.read_i32::<Endian>()?;
        println!("{}: n_mel       = {}\n", function!(), n_mel);
        println!("{}: n_ff   = {}\n", function!(), n_ff);
        let n = (n_mel * n_ff) as usize;
        // let mut filters_data: Vec<f32> = vec![0.0; n];
        // for i in 0..n {
        //     filters_data[i] = r.read_f32::<Endian>()?;
        // }

        let filters_data: Vec<f32> = (0..n).map(|_| r.read_f32::<Endian>().unwrap()).collect();
        println!(
            "{}: filters_data   = {}\n",
            function!(),
            filters_data[n - 1]
        );
        Ok(WhisperFilters {
            n_mel: n_mel,
            n_ff: n_ff,
            data: Arc::new(MthVecF32(UnsafeCell::new(filters_data))),
        })
    }
}

type Id = i32;
type Token = String;

struct WhisperVocab {
    n_vocab: i32,
    token_to_id: HashMap<Token, Id>,
    id_to_token: HashMap<Id, Token>,
    token_eot: Id,
    token_sot: Id,
    token_prev: Id,
    token_solm: Id, // ??
    token_not: Id,  // no timestamps
    token_beg: Id,

    // available tasks
    token_translate: Id,
    token_transcribe: Id,
}

impl Default for WhisperVocab {
    fn default() -> Self {
        WhisperVocab {
            n_vocab: 51864,
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            token_eot: 50256,
            token_sot: 50257,
            token_prev: 50360,
            token_solm: 50361, // ??
            token_not: 50362,  // no timestamps
            token_beg: 50363,

            // available tasks
            token_translate: 50358,
            token_transcribe: 50359,
        }
    }
}

impl WhisperVocab {
    fn load<T: Read>(mut self, n_vocab: i32, r: &mut T) -> WsResult<WhisperVocab> {
        for i in 0..n_vocab {
            let len: u32 = r.read_u32::<Endian>()?;
            let mut tmp = vec![0; len as usize];
            r.read_exact(&mut tmp)?;
            let word = String::from_utf8_lossy(&tmp).to_string();
            if i == 50256 {
                println!("{}: vocab[{}] =       = {}\n", function!(), i, word);
            }
            self.token_to_id.insert(word.clone(), i);
            self.id_to_token.insert(i, word);
        }

        Ok(self)
    }

    const fn is_multilingual(&self) -> bool {
        self.n_vocab == 51865
    }
}

struct WhisperSegment {
    t0: i64,
    t1: i64,
    text: String,
    tokens: Vec<WhisperTokenData>,
}

#[derive(Default)]
struct WhisperHparams {
    n_vocab: i32,
    n_audio_ctx: i32,
    n_audio_state: i32,
    n_audio_head: i32,
    n_audio_layer: i32,
    n_text_ctx: i32,
    n_text_state: i32,
    n_text_head: i32,
    n_text_layer: i32,
    n_mels: i32,
    f16: i32,
}

impl WhisperHparams {
    fn load<T: Read>(r: &mut T) -> WsResult<WhisperHparams> {
        let n_vocab: i32 = r.read_i32::<Endian>()?;
        let n_audio_ctx: i32 = r.read_i32::<Endian>()?;
        let n_audio_state: i32 = r.read_i32::<Endian>()?;
        let n_audio_head: i32 = r.read_i32::<Endian>()?;
        let n_audio_layer: i32 = r.read_i32::<Endian>()?;
        let n_text_ctx: i32 = r.read_i32::<Endian>()?;
        let n_text_state: i32 = r.read_i32::<Endian>()?;
        let n_text_head: i32 = r.read_i32::<Endian>()?;
        let n_text_layer: i32 = r.read_i32::<Endian>()?;
        let n_mels: i32 = r.read_i32::<Endian>()?;
        let f16: i32 = r.read_i32::<Endian>()?;
        println!("{}: n_vocab       = {}\n", function!(), n_vocab);
        println!("{}: n_audio_ctx   = {}\n", function!(), n_audio_ctx);
        println!("{}: n_audio_state = {}\n", function!(), n_audio_state);
        println!("{}: n_audio_head  = {}\n", function!(), n_audio_head);
        println!("{}: n_audio_layer = {}\n", function!(), n_audio_layer);
        println!("{}: n_text_ctx    = {}\n", function!(), n_text_ctx);
        println!("{}: n_text_state  = {}\n", function!(), n_text_state);
        println!("{}: n_text_head   = {}\n", function!(), n_text_head);
        println!("{}: n_text_layer  = {}\n", function!(), n_text_layer);
        println!("{}: n_mels        = {}\n", function!(), n_mels);
        println!("{}: f16           = {}\n", function!(), f16);
        Ok(WhisperHparams {
            n_vocab,
            n_audio_ctx,
            n_audio_state,
            n_audio_head,
            n_audio_layer,
            n_text_ctx,
            n_text_state,
            n_text_head,
            n_text_layer,
            n_mels,
            f16,
        })
    }
}

struct WhisperLayerEncoder {
    // encoder.blocks.*.attn_ln
    attn_ln_0_w: GsTensor,
    attn_ln_0_b: GsTensor,

    // encoder.blocks.*.attn.out
    attn_ln_1_w: GsTensor,
    attn_ln_1_b: GsTensor,

    // encoder.blocks.*.attn.query
    attn_q_w: GsTensor,
    attn_q_b: GsTensor,

    // encoder.blocks.*.attn.key
    attn_k_w: GsTensor,

    // encoder.blocks.*.attn.value
    attn_v_w: GsTensor,
    attn_v_b: GsTensor,

    // encoder.blocks.*.mlp_ln
    mlp_ln_w: GsTensor,
    mlp_ln_b: GsTensor,

    // encoder.blocks.*.mlp.0
    mlp_0_w: GsTensor,
    mlp_0_b: GsTensor,

    // encoder.blocks.*.mlp.2
    mlp_1_w: GsTensor,
    mlp_1_b: GsTensor,
}

struct WhisperLayerDecoder {
    attn_ln_0_w: GsTensor,
    attn_ln_0_b: GsTensor,

    attn_ln_1_w: GsTensor,
    attn_ln_1_b: GsTensor,

    attn_q_w: GsTensor,
    attn_q_b: GsTensor,

    attn_k_w: GsTensor,

    attn_v_w: GsTensor,
    attn_v_b: GsTensor,

    cross_attn_ln_0_w: GsTensor,
    cross_attn_ln_0_b: GsTensor,

    cross_attn_ln_1_w: GsTensor,
    cross_attn_ln_1_b: GsTensor,

    cross_attn_q_w: GsTensor,
    cross_attn_q_b: GsTensor,

    cross_attn_k_w: GsTensor,

    cross_attn_v_w: GsTensor,
    cross_attn_v_b: GsTensor,

    mlp_ln_w: GsTensor,
    mlp_ln_b: GsTensor,

    mlp_0_w: GsTensor,
    mlp_0_b: GsTensor,

    mlp_1_w: GsTensor,
    mlp_1_b: GsTensor,
}

#[derive(Default)]
struct WhisperMel {
    n_len: usize,
    n_mel: usize,
    data: Arc<MthVecF32>,
}

impl WhisperMel {
    fn new() -> WhisperMel {
        WhisperMel {
            n_len: 0,
            n_mel: 0,
            data: Arc::new(MthVecF32(UnsafeCell::new(Vec::new()))),
        }
    }
}

unsafe impl Send for MthVecF32 {}
unsafe impl Sync for MthVecF32 {}

#[derive(Default)]
struct MthVecF32(UnsafeCell<Vec<f32>>);

impl MthVecF32 {
    pub(crate) unsafe fn borrow(&self) -> &Vec<f32> {
        &*self.0.get()
    }
    pub(crate) unsafe fn borrow_mut(&self) -> &mut Vec<f32> {
        &mut *self.0.get()
    }
}

// 打开文件流
fn open_file_stream(fname: &str) -> WsResult<BufReader<File>> {
    let file = File::open(fname)?;
    let buf_reader = BufReader::new(file);
    Ok(buf_reader)
}

struct WhisperModel {
    mtype: EModel,
    hparams: WhisperHparams,
    filters: WhisperFilters,

    e_pe: GsTensor,
    // encoder.conv1
    e_conv_1_w: GsTensor,
    e_conv_1_b: GsTensor,
    // encoder.conv2
    e_conv_2_w: GsTensor,
    e_conv_2_b: GsTensor,
    // encoder.ln_post
    e_ln_w: GsTensor,
    e_ln_b: GsTensor,
    // decoder.positional_embedding
    d_pe: GsTensor,
    // decoder.token_embedding
    d_te: GsTensor,
    // decoder.ln
    d_ln_w: GsTensor,
    d_ln_b: GsTensor,

    layers_encoder: Vec<WhisperLayerEncoder>,
    layers_decoder: Vec<WhisperLayerDecoder>,

    memory_k: GsTensor,
    memory_v: GsTensor,

    memory_cross_k: GsTensor,
    memory_cross_v: GsTensor,

    // kv_memory: Box<[GsTensor; 4]>,
    n_loaded: usize,
}

impl WhisperModel {
    fn load<T: Read + BufRead>(
        r: &mut T,
        mtype: EModel,
        hparams: WhisperHparams,
        filters: WhisperFilters,
        buf_model: &[u8],
        buf_memory: &[u8],
    ) -> WsResult<WhisperModel> {
        let wtype = if hparams.f16 == 1 {
            DType::F16
        } else {
            DType::F32
        };

        let mut ctx_size = 0usize;
        let mut ctx_mem_size = 0usize;
        {
            let n_vocab = hparams.n_vocab as usize;
            let n_audio_ctx = hparams.n_audio_ctx as usize;
            let n_audio_state = hparams.n_audio_state as usize;
            let n_audio_layer = hparams.n_audio_layer as usize;

            let n_text_ctx = hparams.n_text_ctx as usize;
            let n_text_state = hparams.n_text_state as usize;
            let n_text_layer = hparams.n_text_layer as usize;
            let n_mels = hparams.n_mels as usize;
            // encoder
            {
                ctx_size += n_audio_ctx * n_audio_state * get_type_size(DType::F32); // e_pe;

                ctx_size += 3 * n_mels * n_audio_state * get_type_size(wtype); // e_conv_1_w
                ctx_size += n_audio_state * get_type_size(DType::F32); // e_conv_1_b

                ctx_size += 3 * n_audio_state * n_audio_state * get_type_size(wtype); // e_conv_2_w
                ctx_size += n_audio_state * get_type_size(DType::F32); // e_conv_2_b

                ctx_size += n_audio_state * get_type_size(DType::F32); // e_ln_w;
                ctx_size += n_audio_state * get_type_size(DType::F32);
            }

            // decoder
            {
                // TODO: F16 .. maybe not?
                ctx_size += n_text_ctx * n_text_state * get_type_size(DType::F32); // d_pe;

                ctx_size += n_vocab * n_text_state * get_type_size(wtype.clone()); // d_te;

                ctx_size += n_text_state * get_type_size(DType::F32); // d_ln_w;
                ctx_size += n_text_state * get_type_size(DType::F32);
                // d_ln_b;
            }

            // encoder layers
            {
                ctx_size += n_audio_layer * (n_audio_state * get_type_size(DType::F32)); // mlp_ln_w
                ctx_size += n_audio_layer * (n_audio_state * get_type_size(DType::F32)); // mlp_ln_b

                ctx_size +=
                    n_audio_layer * (4 * n_audio_state * n_audio_state * get_type_size(wtype)); // mlp_0_w
                ctx_size += n_audio_layer * (4 * n_audio_state * get_type_size(DType::F32)); // mlp_0_b

                ctx_size +=
                    n_audio_layer * (4 * n_audio_state * n_audio_state * get_type_size(wtype)); // mlp_1_w
                ctx_size += n_audio_layer * (n_audio_state * get_type_size(DType::F32)); // mlp_1_b

                ctx_size += n_audio_layer * (n_audio_state * get_type_size(DType::F32)); // attn_ln_0_w
                ctx_size += n_audio_layer * (n_audio_state * get_type_size(DType::F32)); // attn_ln_0_b

                ctx_size += n_audio_layer * (n_audio_state * n_audio_state * get_type_size(wtype)); // attn_q_w
                ctx_size += n_audio_layer * (n_audio_state * get_type_size(DType::F32)); // attn_q_b

                ctx_size += n_audio_layer * (n_audio_state * n_audio_state * get_type_size(wtype)); // attn_k_w

                ctx_size += n_audio_layer * (n_audio_state * n_audio_state * get_type_size(wtype)); // attn_v_w
                ctx_size += n_audio_layer * (n_audio_state * get_type_size(DType::F32)); // attn_v_b

                ctx_size += n_audio_layer * (n_audio_state * n_audio_state * get_type_size(wtype)); // attn_ln_1_w
                ctx_size += n_audio_layer * (n_audio_state * get_type_size(DType::F32));
                // attn_ln_1_b
            }

            // decoder layers
            {
                ctx_size += n_text_layer * (n_text_state * get_type_size(DType::F32)); // mlp_ln_w
                ctx_size += n_text_layer * (n_text_state * get_type_size(DType::F32)); // mlp_ln_b

                ctx_size += n_text_layer * (4 * n_text_state * n_text_state * get_type_size(wtype)); // mlp_0_w
                ctx_size += n_text_layer * (4 * n_text_state * get_type_size(DType::F32)); // mlp_0_b

                ctx_size += n_text_layer * (4 * n_text_state * n_text_state * get_type_size(wtype)); // mlp_1_w
                ctx_size += n_text_layer * (n_text_state * get_type_size(DType::F32)); // mlp_1_b

                ctx_size += n_text_layer * (n_text_state * get_type_size(DType::F32)); // attn_ln_0_w
                ctx_size += n_text_layer * (n_text_state * get_type_size(DType::F32)); // attn_ln_0_b

                ctx_size += n_text_layer * (n_text_state * n_text_state * get_type_size(wtype)); // attn_q_w
                ctx_size += n_text_layer * (n_text_state * get_type_size(DType::F32)); // attn_q_b

                ctx_size += n_text_layer * (n_text_state * n_text_state * get_type_size(wtype)); // attn_k_w

                ctx_size += n_text_layer * (n_text_state * n_text_state * get_type_size(wtype)); // attn_v_w
                ctx_size += n_text_layer * (n_text_state * get_type_size(DType::F32)); // attn_v_b

                ctx_size += n_text_layer * (n_text_state * n_text_state * get_type_size(wtype)); // attn_ln_1_w
                ctx_size += n_text_layer * (n_text_state * get_type_size(DType::F32)); // attn_ln_1_b
                                                                                       //
                ctx_size += n_text_layer * (n_text_state * get_type_size(DType::F32)); // cross_attn_ln_0_w
                ctx_size += n_text_layer * (n_text_state * get_type_size(DType::F32)); // cross_attn_ln_0_b

                ctx_size += n_text_layer * (n_text_state * n_text_state * get_type_size(wtype)); // cross_attn_q_w
                ctx_size += n_text_layer * (n_text_state * get_type_size(DType::F32)); // cross_attn_q_b

                ctx_size += n_text_layer * (n_text_state * n_text_state * get_type_size(wtype)); // cross_attn_k_w

                ctx_size += n_text_layer * (n_text_state * n_text_state * get_type_size(wtype)); // cross_attn_v_w
                ctx_size += n_text_layer * (n_text_state * get_type_size(DType::F32)); // cross_attn_v_b

                ctx_size += n_text_layer * (n_text_state * n_text_state * get_type_size(wtype)); // cross_attn_ln_1_w
                ctx_size += n_text_layer * (n_text_state * get_type_size(DType::F32));
                // cross_attn_ln_1_b
            }

            ctx_mem_size += n_text_layer * n_text_ctx * n_text_state * get_type_size(DType::F16); // memory_k
            ctx_mem_size += n_text_layer * n_text_ctx * n_text_state * get_type_size(DType::F16); // memory_v

            ctx_mem_size += n_text_layer * n_audio_ctx * n_text_state * get_type_size(DType::F16); // memory_cross_k
            ctx_mem_size += n_text_layer * n_audio_ctx * n_text_state * get_type_size(DType::F16); // memory_cross_v

            ctx_size += (15 + 15 * n_audio_layer + 24 * n_text_layer) * 256; // object overhead

            println!(
                "{}: ggml ctx size = {:7.2}  MB\n",
                function!(),
                ctx_size as f32 / (1024.0 * 1024.0),
            );
        }
        let mut tensors: HashMap<String, *mut GsTensor> = HashMap::new();

        let mut whisper_mode = {
            let mut tensor_ctx = TensorContext::new(buf_model);
            let n_vocab = hparams.n_vocab as usize;
            let n_audio_ctx = hparams.n_audio_ctx as usize;
            let n_audio_state = hparams.n_audio_state as usize;
            let n_audio_layer = hparams.n_audio_layer as usize;

            let n_text_ctx = hparams.n_text_ctx as usize;
            let n_text_state = hparams.n_text_state as usize;
            let n_text_layer = hparams.n_text_layer as usize;
            let n_mels = hparams.n_mels as usize;

            // encoder
            let mut e_pe = new_tensor_2d(&mut tensor_ctx, DType::F32, n_audio_state, n_audio_ctx)?;
            let mut e_conv_1_w = new_tensor_3d(&mut tensor_ctx, wtype, 3, n_mels, n_audio_state)?;
            let mut e_conv_1_b = new_tensor_2d(&mut tensor_ctx, DType::F32, 1, n_audio_state)?;

            let mut e_conv_2_w =
                new_tensor_3d(&mut tensor_ctx, wtype, 3, n_audio_state, n_audio_state)?;
            let mut e_conv_2_b = new_tensor_2d(&mut tensor_ctx, DType::F32, 1, n_audio_state)?;

            let mut e_ln_w = new_tensor_1d(&mut tensor_ctx, DType::F32, n_audio_state)?;
            let mut e_ln_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_audio_state)?;

            // let weights = Box::new([
            //     e_pe, e_conv_1_w, e_conv_1_b, e_conv_2_w, e_conv_2_b, e_ln_w, e_ln_b, d_pe, d_te,
            //     d_ln_w, d_ln_b,
            // ]);
            tensors.insert(
                "encoder.positional_embedding".to_string(),
                &mut e_pe as *mut GsTensor,
            );
            tensors.insert(
                "encoder.conv1.weight".to_string(),
                &mut e_conv_1_w as *mut GsTensor,
            );
            tensors.insert(
                "encoder.conv1.bias".to_string(),
                &mut e_conv_1_b as *mut GsTensor,
            );
            tensors.insert(
                "encoder.conv2.weight".to_string(),
                &mut e_conv_2_w as *mut GsTensor,
            );
            tensors.insert(
                "encoder.conv2.bias".to_string(),
                &mut e_conv_2_b as *mut GsTensor,
            );
            tensors.insert(
                "encoder.ln_post.weight".to_string(),
                &mut e_ln_w as *mut GsTensor,
            );
            tensors.insert(
                "encoder.ln_post.bias".to_string(),
                &mut e_ln_b as *mut GsTensor,
            );

            let mut layers_encoder: Vec<WhisperLayerEncoder> = Vec::with_capacity(n_audio_layer);
            let mut layers_decoder: Vec<WhisperLayerDecoder> = Vec::with_capacity(n_text_layer);
            for i in 0..n_audio_layer {
                let mlp_ln_w = new_tensor_1d(&mut tensor_ctx, DType::F32, n_audio_state)?;
                let mlp_ln_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_audio_state)?;
                let mlp_0_w =
                    new_tensor_2d(&mut tensor_ctx, wtype, n_audio_state, 4 * n_audio_state)?;
                let mlp_0_b = new_tensor_1d(&mut tensor_ctx, DType::F32, 4 * n_audio_state)?;
                let mlp_1_w =
                    new_tensor_2d(&mut tensor_ctx, wtype, 4 * n_audio_state, n_audio_state)?;
                let mlp_1_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_audio_state)?;

                let attn_ln_0_w = new_tensor_1d(&mut tensor_ctx, DType::F32, n_audio_state)?;
                let attn_ln_0_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_audio_state)?;
                let attn_q_w = new_tensor_2d(&mut tensor_ctx, wtype, n_audio_state, n_audio_state)?;
                let attn_q_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_audio_state)?;

                let attn_k_w = new_tensor_2d(&mut tensor_ctx, wtype, n_audio_state, n_audio_state)?;

                let attn_v_w = new_tensor_2d(&mut tensor_ctx, wtype, n_audio_state, n_audio_state)?;
                let attn_v_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_audio_state)?;

                let attn_ln_1_w =
                    new_tensor_2d(&mut tensor_ctx, wtype, n_audio_state, n_audio_state)?;
                let attn_ln_1_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_audio_state)?;

                layers_encoder.push(WhisperLayerEncoder {
                    // encoder.blocks.*.attn_ln
                    attn_ln_0_w,
                    attn_ln_0_b,

                    // encoder.blocks.*.attn.out
                    attn_ln_1_w,
                    attn_ln_1_b,

                    // encoder.blocks.*.attn.query
                    attn_q_w,
                    attn_q_b,

                    // encoder.blocks.*.attn.key
                    attn_k_w,

                    // encoder.blocks.*.attn.value
                    attn_v_w,
                    attn_v_b,

                    // encoder.blocks.*.mlp_ln
                    mlp_ln_w,
                    mlp_ln_b,

                    // encoder.blocks.*.mlp.0
                    mlp_0_w,
                    mlp_0_b,

                    // encoder.blocks.*.mlp.2
                    mlp_1_w,
                    mlp_1_b,
                });

                let encoder = layers_encoder.last_mut().unwrap();

                tensors.insert(
                    format!("encoder.blocks.{}.mlp_ln.weight", i),
                    &mut encoder.mlp_ln_w as *mut GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.mlp_ln.bias", i),
                    &mut encoder.mlp_ln_b as *mut GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.mlp.0.weight", i),
                    &mut encoder.mlp_0_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("encoder.blocks.{}.mlp.0.bias", i),
                    &mut encoder.mlp_0_b as *mut GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.mlp.2.weight", i),
                    &mut encoder.mlp_1_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("encoder.blocks.{}.mlp.2.bias", i),
                    &mut encoder.mlp_1_b as *mut GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn_ln.weight", i),
                    &mut encoder.attn_ln_0_w as *mut GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn_ln.bias", i),
                    &mut encoder.attn_ln_0_b as *mut GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn.query.weight", i),
                    &mut encoder.attn_q_w as *mut GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn.query.bias", i),
                    &mut encoder.attn_q_b as *mut GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn.key.weight", i),
                    &mut encoder.attn_k_w as *mut GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn.value.weight", i),
                    &mut encoder.attn_v_w as *mut GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn.value.bias", i),
                    &mut encoder.attn_v_b as *mut GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn.out.weight", i),
                    &mut encoder.attn_ln_1_w as *mut GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn.out.bias", i),
                    &mut encoder.attn_ln_1_b as *mut GsTensor,
                );
            }

            let mut d_pe = new_tensor_2d(&mut tensor_ctx, DType::F32, n_text_state, n_text_ctx)?;

            let mut d_te = new_tensor_2d(&mut tensor_ctx, wtype, n_text_state, n_vocab)?;
            let mut d_ln_w = new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;
            let mut d_ln_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;

            tensors.insert(
                "decoder.positional_embedding".to_string(),
                &mut d_pe as *mut GsTensor,
            );
            tensors.insert(
                "decoder.token_embedding.weight".to_string(),
                &mut d_te as *mut GsTensor,
            );
            tensors.insert(
                "decoder.ln.weight".to_string(),
                &mut d_ln_w as *mut GsTensor,
            );
            tensors.insert("decoder.ln.bias".to_string(), &mut d_ln_b as *mut GsTensor);

            for i in 0..n_text_layer {
                let mut mlp_ln_w = new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;
                let mut mlp_ln_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;

                let mut mlp_0_w =
                    new_tensor_2d(&mut tensor_ctx, wtype, n_text_state, 4 * n_text_state)?;
                let mut mlp_0_b = new_tensor_1d(&mut tensor_ctx, DType::F32, 4 * n_text_state)?;

                let mut mlp_1_w =
                    new_tensor_2d(&mut tensor_ctx, wtype, 4 * n_text_state, n_text_state)?;
                let mut mlp_1_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;

                let mut attn_ln_0_w = new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;
                let mut attn_ln_0_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;

                let mut attn_q_w =
                    new_tensor_2d(&mut tensor_ctx, wtype, n_text_state, n_text_state)?;
                let mut attn_q_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;

                let mut attn_k_w =
                    new_tensor_2d(&mut tensor_ctx, wtype, n_text_state, n_text_state)?;

                let mut attn_v_w =
                    new_tensor_2d(&mut tensor_ctx, wtype, n_text_state, n_text_state)?;
                let mut attn_v_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;

                let mut attn_ln_1_w =
                    new_tensor_2d(&mut tensor_ctx, wtype, n_text_state, n_text_state)?;
                let mut attn_ln_1_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;

                let mut cross_attn_ln_0_w =
                    new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;
                let mut cross_attn_ln_0_b =
                    new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;

                let mut cross_attn_q_w =
                    new_tensor_2d(&mut tensor_ctx, wtype, n_text_state, n_text_state)?;
                let mut cross_attn_q_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;

                let mut cross_attn_k_w =
                    new_tensor_2d(&mut tensor_ctx, wtype, n_text_state, n_text_state)?;

                let mut cross_attn_v_w =
                    new_tensor_2d(&mut tensor_ctx, wtype, n_text_state, n_text_state)?;
                let mut cross_attn_v_b = new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;

                let mut cross_attn_ln_1_w =
                    new_tensor_2d(&mut tensor_ctx, wtype, n_text_state, n_text_state)?;
                let mut cross_attn_ln_1_b =
                    new_tensor_1d(&mut tensor_ctx, DType::F32, n_text_state)?;

                layers_decoder.push(WhisperLayerDecoder {
                    attn_ln_0_w,
                    attn_ln_0_b,
                    attn_ln_1_w,
                    attn_ln_1_b,
                    attn_q_w,
                    attn_q_b,
                    attn_k_w,
                    attn_v_w,
                    attn_v_b,
                    cross_attn_ln_0_w,
                    cross_attn_ln_0_b,
                    cross_attn_ln_1_w,
                    cross_attn_ln_1_b,
                    cross_attn_q_w,
                    cross_attn_q_b,
                    cross_attn_k_w,
                    cross_attn_v_w,
                    cross_attn_v_b,
                    mlp_ln_w,
                    mlp_ln_b,
                    mlp_0_w,
                    mlp_0_b,
                    mlp_1_w,
                    mlp_1_b,
                });
                let decoder = layers_decoder.last_mut().unwrap();

                tensors.insert(
                    format!("decoder.blocks.{}.mlp_ln.weight", i),
                    &mut decoder.mlp_ln_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.mlp_ln.bias", i),
                    &mut decoder.mlp_ln_b as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.mlp.0.weight", i),
                    &mut decoder.mlp_0_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.mlp.0.bias", i),
                    &mut decoder.mlp_0_b as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.mlp.2.weight", i),
                    &mut decoder.mlp_1_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.mlp.2.bias", i),
                    &mut decoder.mlp_1_b as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn_ln.weight", i),
                    &mut decoder.attn_ln_0_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn_ln.bias", i),
                    &mut decoder.attn_ln_0_b as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn.query.weight", i),
                    &mut decoder.attn_q_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn.query.bias", i),
                    &mut decoder.attn_q_b as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn.key.weight", i),
                    &mut decoder.attn_k_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn.value.weight", i),
                    &mut decoder.attn_v_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn.value.bias", i),
                    &mut decoder.attn_v_b as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn.out.weight", i),
                    &mut decoder.attn_ln_1_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn.out.bias", i),
                    &mut decoder.attn_ln_1_b as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn_ln.weight", i),
                    &mut decoder.cross_attn_ln_0_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn_ln.bias", i),
                    &mut decoder.cross_attn_ln_0_b as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn.query.weight", i),
                    &mut decoder.cross_attn_q_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn.query.bias", i),
                    &mut decoder.cross_attn_q_b as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn.key.weight", i),
                    &mut decoder.cross_attn_k_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn.value.weight", i),
                    &mut decoder.cross_attn_v_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn.value.bias", i),
                    &mut decoder.cross_attn_v_b as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn.out.weight", i),
                    &mut decoder.cross_attn_ln_1_w as *mut GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn.out.bias", i),
                    &mut decoder.cross_attn_ln_1_b as *mut GsTensor,
                );
            }

            //load kv memmory
            let mut tensor_ctx: TensorContext = TensorContext::new(buf_memory);

            let n_text_ctx = hparams.n_text_ctx as usize;
            let n_text_state = hparams.n_text_state as usize;
            let n_text_layer = hparams.n_text_layer as usize;

            let n_mem = n_text_layer * n_text_ctx;
            let n_elements = n_text_state * n_mem;

            let memory_k = new_tensor_1d(&mut tensor_ctx, DType::F16, n_elements)?;
            let memory_v = new_tensor_1d(&mut tensor_ctx, DType::F16, n_elements)?;

            let n_audio_ctx = hparams.n_audio_ctx as usize;

            let n_mem = n_text_layer * n_audio_ctx;
            let n_elements = n_text_state * n_mem;
            let memory_cross_k = new_tensor_1d(&mut tensor_ctx, DType::F16, n_elements)?;
            let memory_cross_v = new_tensor_1d(&mut tensor_ctx, DType::F16, n_elements)?;

            WhisperModel {
                mtype: mtype,
                hparams: hparams,
                filters: filters,
                e_pe,
                e_conv_1_w,
                e_conv_1_b,
                e_conv_2_w,
                e_conv_2_b,
                e_ln_w,
                e_ln_b,
                d_pe,
                d_te,
                d_ln_w,
                d_ln_b,
                layers_encoder,
                layers_decoder,
                memory_k,
                memory_v,
                memory_cross_k,
                memory_cross_v,
                n_loaded: 1,
            }
        };
        // load weights
        {
            let mut total_size: usize = 0;
            let j: usize = 0;
            loop {
                let n_dims = r.read_i32::<Endian>()?;
                let length = r.read_i32::<Endian>()?;
                let ftype = r.read_i32::<Endian>()?;

                let mut nelements: usize = 1;
                let mut ne: [usize; 3] = [1, 1, 1];
                // let n_dims = 3; // Assume this value is set appropriately
                // println!("n_dims:{}", n_dims);
                for i in 0..n_dims as usize {
                    ne[i] = r.read_i32::<Endian>()? as usize;
                    nelements *= ne[i];
                }
                //  println!("nelements:{}", nelements);
                let mut buffer = vec![0; length as usize];
                r.read_exact(&mut buffer)?;
                let name = String::from_utf8_lossy(&buffer).to_string();
                let ref_tensor = tensors
                    .get_mut(name.as_str())
                    .ok_or(WsError::UnknownTensor(name.clone()))?;

                if let Some(tensor) = unsafe { (*ref_tensor).as_mut() } {
                    if tensor.elem_count() != nelements {
                        return Err(WsError::WrongSizeTensor(
                            name,
                            tensor.elem_count(),
                            nelements,
                        ));
                    }
                    let shape = tensor.dim().shape();
                    for i in 0..shape.len() {
                        if shape[i] != ne[i] {
                            return Err(WsError::WrongShapeTensor(
                                name,
                                shape.to_vec(),
                                ne.to_vec(),
                            ));
                        }
                    }
                    let bpe = if ftype == 0 {
                        std::mem::size_of::<f32>()
                    } else {
                        std::mem::size_of::<galois::F16>()
                    };
                    if nelements * bpe != tensor.nbytes() {
                        return Err(WsError::WrongBytesTensor(
                            name,
                            tensor.nbytes(),
                            nelements * bpe,
                        ));
                    }
                    {
                        // let mut x = vec![0u8; tensor.as_bytes_mut().len()];
                        r.read_exact(tensor.as_bytes_mut())?;
                    }
                    // if name == "encoder.blocks.0.attn.query.weight".to_string() {
                    //     let x: &[F16] = unsafe { tensor.as_slice::<F16>() };
                    //     let mut sum: f64 = 0.0;
                    //     for i in 0..x.len() {
                    //         sum += x[i].abs().to_f64();
                    //         if i < 10 || i > tensor.elem_count() - 10 {
                    //             print!("{:?},", x[i])
                    //         }
                    //     }
                    //     println!(
                    //         "encoder.blocks.0.attn.query.weight:{:?},shape:{:?},stride:{:?}",
                    //         sum,
                    //         tensor.shape(),
                    //         tensor.dim().stride_4d()
                    //     );
                    // }

                    // println!("name:{},{}", name, tensor.as_bytes_mut().len());

                    total_size += tensor.nbytes();
                    whisper_mode.n_loaded += 1;
                    match r.fill_buf() {
                        Ok(r) => {
                            if r.len() < 12 {
                                break;
                            }
                        }
                        Err(e) => match e.kind() {
                            std::io::ErrorKind::UnexpectedEof => break,
                            _ => return Err(WsError::UnexpectIO(e)),
                        },
                    }
                } else {
                    println!("break");
                    return Err(WsError::BadRefTensor(name));
                }
            }
            println!(
                "{}: model size    = {:7.2} MB",
                function!(),
                total_size as f32 / 1024.0 / 1024.0
            );
        }
        Ok(whisper_mode)
    }
}

//离散傅里叶变换
fn dft(inp: &[f32], out: &mut Vec<f32>) {
    let n = inp.len();
    out.resize(n * 2, 0.0);

    for k in 0..n {
        let mut re = 0.0;
        let mut im = 0.0;
        for n_val in 0..n {
            let angle = 2.0 * std::f32::consts::PI * (k * n_val) as f32 / n as f32;
            re += inp[n_val] * angle.cos();
            im -= inp[n_val] * angle.sin();
        }
        out[k * 2] = re;
        out[k * 2 + 1] = im;
    }
}

//快速傅里叶变换
fn fft(inp: &[f32], out: &mut Vec<f32>) {
    out.resize(inp.len() * 2, 0.0);
    let n = inp.len();
    if n == 1 {
        out[0] = inp[0];
        out[1] = 0.0;
        return;
    }

    if n % 2 == 1 {
        dft(inp, out);
        return;
    }

    let mut even: Vec<f32> = Vec::with_capacity(n / 2);
    let mut odd: Vec<f32> = Vec::with_capacity(n / 2);

    for i in 0..n {
        if i % 2 == 0 {
            even.push(inp[i]);
        } else {
            odd.push(inp[i]);
        }
    }

    let mut even_fft: Vec<f32> = vec![0.0; n];
    let mut odd_fft: Vec<f32> = vec![0.0; n];

    fft(&even, &mut even_fft);
    fft(&odd, &mut odd_fft);

    for k in 0..n / 2 {
        let theta = 2.0 * std::f32::consts::PI * (k as f32) / (n as f32);

        let re = theta.cos();
        let im = -theta.sin();

        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];

        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + n / 2)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + n / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

// 转成梅尔频谱
fn log_mel_spectrogram(
    samples: Arc<Vec<f32>>,
    sameple_rate: usize,
    fft_size: usize,
    fft_step: usize,
    n_mel: usize,
    n_threads: usize, //sped
    filters: &WhisperFilters,
    speed_up: bool,
    mel: &mut WhisperMel,
) -> WsResult<()> {
    let n_samples = samples.len();
    let fft_size_f32 = fft_size as f32;
    let _hann: Vec<f32> = (0..fft_size)
        .map(|i| 0.5 * (1.0 - ((2.0 * std::f32::consts::PI * i as f32) / fft_size_f32).cos()))
        .collect();

    let y: f32 = _hann.iter().sum();
    println!("_hann:{:?}", y);
    let mhann = Arc::new(_hann);
    mel.n_mel = n_mel;
    let n_len = n_samples / fft_step;
    mel.n_len = n_len;
    unsafe {
        mel.data.borrow_mut().resize(mel.n_mel * mel.n_len, 0.0f32);
        println!("mel.data.len:{}", mel.data.borrow().len());
        let n_fft = 1 + (if speed_up { fft_size / 4 } else { fft_size / 2 });
        let mut works: Vec<JoinHandle<()>> = Vec::with_capacity(n_threads);
        for iw in 0..n_threads {
            let mel_data = mel.data.clone();
            let m_samples = samples.clone();
            let filter_data = filters.data.clone();
            let hann = mhann.clone();
            let work = std::thread::spawn(move || {
                let data: &mut Vec<f32> = mel_data.borrow_mut();
                let filter_data = filter_data.borrow();
                let ith = iw;
                let mut fft_in = vec![0.0f32; fft_size];
                let mut fft_out = vec![0.0f32; 2 * fft_size];
                for i in (ith..n_len).step_by(n_threads) {
                    let offset = i * fft_step;
                    for j in 0..fft_size {
                        if (offset + j < n_samples) {
                            fft_in[j] = hann[j] * m_samples[offset + j];
                        } else {
                            fft_in[j] = 0.0;
                        }
                    }
                    fft(&fft_in, &mut fft_out);
                    for j in 0..fft_size {
                        fft_out[j] = fft_out[2 * j] * fft_out[2 * j]
                            + fft_out[2 * j + 1] * fft_out[2 * j + 1];
                    }

                    for j in 1..fft_size / 2 {
                        fft_out[j] += fft_out[fft_size - j];
                    }

                    if speed_up {
                        // Scale down in the frequency domain results in a speed up in the time domain
                        for j in 0..n_fft {
                            fft_out[j] = 0.5 * (fft_out[2 * j] + fft_out[2 * j + 1]);
                        }
                    }

                    // Mel spectrogram
                    for j in 0..n_mel {
                        let mut sum = 0.0;

                        for k in 0..n_fft {
                            sum += fft_out[k] * filter_data[j * n_fft + k];
                        }

                        if sum < 1e-10 {
                            sum = 1e-10;
                        }

                        sum = sum.log10();

                        data[j * n_len + i] = sum;
                    }
                }

                println!("{}", iw);
            });
            works.push(work);
        }

        for v in works {
            v.join().unwrap();
        }
        let data = mel.data.borrow();
        let x: f32 = data.iter().sum();
        println!("x1:{:?}", x);
        clamp_and_normalize(mel.data.borrow_mut());
    }

    Ok(())
}

fn clamp_and_normalize(mel: &mut Vec<f32>) {
    let mut mmax = -1e20;
    for value in mel.iter() {
        if *value as f64 > mmax {
            mmax = *value as f64;
        }
    }

    mmax -= 8.0;

    for value in mel.iter_mut() {
        if (*value as f64) < mmax {
            *value = mmax as f32;
        }

        *value = (*value + 4.0) / 4.0;
    }
}

fn convert_integer_to_float_audio(samples: &[i16]) -> Vec<f32> {
    let mut floats = Vec::with_capacity(samples.len());
    for sample in samples {
        floats.push(*sample as f32 / 32768.0);
    }
    floats
}

fn whisper_pcm_to_mel(ctx: &mut WhisperContext, samples: Arc<Vec<f32>>) -> WsResult<()> {
    let mut x: f32 = 0.0;
    for i in 0..samples.len() {
        x += samples[i];
    }
    println!("y:{}", x);

    let data = unsafe { &ctx.model.filters.data.borrow() };
    let x: f32 = data.iter().sum();
    println!("filters:{:?}", x);

    log_mel_spectrogram(
        samples,
        WHISPER_SAMPLE_RATE,
        WHISPER_N_FFT,
        WHISPER_HOP_LENGTH,
        WHISPER_N_MEL,
        4,
        &ctx.model.filters,
        false,
        &mut ctx.mel,
    )?;
    // let data = unsafe { ctx.mel.data.borrow() };
    // let x: f32 = data.iter().sum();
    // println!("x:{:?}", x);
    Ok(())
}

fn conv_1d_1s(ctx: &mut TensorContext, kernel: &GsTensor, src: &GsTensor) -> WsResult<GsTensor> {
    let ne = [src.shape()[0], kernel.shape()[2]];
    let mut dst = new_tensor(ctx, 2, DType::F32, Shape::from_array(ne))?;
    galois::op::galois_conv_1d_1s(kernel, src, &mut dst)?;
    Ok(dst)
}

fn conv_1d_2s(ctx: &mut TensorContext, kernel: &GsTensor, src: &GsTensor) -> WsResult<GsTensor> {
    let ne = [src.shape()[0] / 2, kernel.shape()[2]];
    let mut dst = new_tensor(ctx, 2, DType::F32, Shape::from_array(ne))?;
    galois::op::galois_conv_1d_2s(kernel, src, &mut dst)?;
    Ok(dst)
}

fn repeat(ctx: &mut TensorContext, src: &GsTensor, cur: &GsTensor) -> WsResult<GsTensor> {
    let mut dst = new_tensor(
        ctx,
        cur.n_dims(),
        src.dtype(),
        Shape::from_slice(cur.shape()),
    )?;
    galois::op::galois_repeat(src, &mut dst)?;
    Ok(dst)
}

fn cpy(src: &GsTensor, cur: &GsTensor) -> WsResult<GsTensor> {
    let mut dst = cur.view();
    galois::op::galois_cpy(src, &mut dst)?;
    Ok(dst)
}

fn add(ctx: &mut TensorContext, a: &GsTensor, b: &GsTensor) -> WsResult<GsTensor> {
    let mut dst = dup_tensor(ctx, a)?;
    galois::op::galois_add(a, b, &mut dst)?;
    Ok(dst)
}

fn scale(ctx: &mut TensorContext, a: &GsTensor, b: &GsTensor) -> WsResult<GsTensor> {
    let mut dst = a.view();
    galois::op::galois_scale(a, b, &mut dst)?;
    Ok(dst)
}

fn matmul(ctx: &mut TensorContext, a: &GsTensor, b: &GsTensor) -> WsResult<GsTensor> {
    let ne = [
        a.ggml_shape()[1],
        b.ggml_shape()[1],
        a.ggml_shape()[2],
        b.ggml_shape()[3],
    ];
    let mut dst = new_tensor(
        ctx,
        std::cmp::min(a.n_dims(), b.n_dims()),
        DType::F32,
        Shape::from_array(ne),
    )?;
    galois::op::galois_matmul(a, b, &mut dst)?;
    Ok(dst)
}

fn mul(ctx: &mut TensorContext, a: &GsTensor, b: &GsTensor) -> WsResult<GsTensor> {
    let mut dst = dup_tensor(ctx, a)?;
    galois::op::galois_mul(a, b, &mut dst)?;
    Ok(dst)
}

fn gelu(ctx: &mut TensorContext, a: &GsTensor) -> WsResult<GsTensor> {
    let mut dst = dup_tensor(ctx, a)?;
    galois::op::galois_gelu(a, &mut dst)?;
    Ok(dst)
}

fn norm(ctx: &mut TensorContext, a: &GsTensor) -> WsResult<GsTensor> {
    let mut dst = dup_tensor(ctx, a)?;
    galois::op::galois_norm(a, &mut dst)?;
    Ok(dst)
}

fn flash_attn(
    ctx: &mut TensorContext,
    q: &GsTensor,
    k: &GsTensor,
    v: &GsTensor,
) -> WsResult<GsTensor> {
    //assert!()
    let mut dst = new_tensor(ctx, 4, DType::F32, Shape::from_slice(q.shape()))?;
    galois::op::galois_flash_attn(q, k, v, &mut dst)?;
    Ok(dst)
}

fn whisper_encode(wctx: &mut WhisperContext, n_threads: usize, mel_offset: usize) -> WsResult<()> {
    let model = &wctx.model;
    let mel_inp = &wctx.mel;
    let hparams = &model.hparams;
    let n_ctx = if wctx.exp_n_audio_ctx > 0 {
        wctx.exp_n_audio_ctx as usize
    } else {
        hparams.n_audio_ctx as usize
    };
    let n_state = hparams.n_audio_state as usize;
    let n_head = hparams.n_audio_head as usize;
    let n_layer = hparams.n_audio_layer as usize;

    let n_mels = hparams.n_mels as usize;
    assert!(mel_inp.n_mel == n_mels as usize);
    let buf_compute = &wctx.buf_compute;
    let mut ctx0: TensorContext = TensorContext::new(buf_compute);
    let mut mel = new_tensor_2d(&mut ctx0, DType::F32, 2 * n_ctx, n_mels)?;
    assert!(mel.dtype() == DType::F32);
    {
        let dst = unsafe { mel.as_slice_mut::<f32>() };
        dst.fill(0.0);

        let i0 = std::cmp::min(mel_offset, mel_inp.n_len);
        let i1: usize = std::cmp::min(mel_offset + 2 * n_ctx, mel_inp.n_len);
        let data = unsafe { mel_inp.data.borrow() };
        for j in 0..mel_inp.n_mel {
            for i in i0..i1 {
                dst[j * 2 * n_ctx + (i - i0)] = data[j * mel_inp.n_len + i];
            }
        }

        let y: f32 = dst.iter().sum();
        println!("y:{:?}", y);
    }
    let mut cur = conv_1d_1s(&mut ctx0, &model.e_conv_1_w, &mel)?;

    // let x: &[f32] = unsafe { cur.as_slice::<f32>() };
    // let mut sum: f64 = 0.0;
    // for i in 0..cur.elem_count() {
    //     sum += x[i].abs() as f64;
    // }

    // println!(
    //     "cur,sum:{:?},sha
    //     pe:{:?},stride:{:?}",
    //     sum,
    //     cur.ggml_shape(),
    //     cur.dim().stride_4d()
    // );
    // return Ok(());

    let mut tmp = repeat(&mut ctx0, &model.e_conv_1_b, &cur)?;

    cur = add(&mut ctx0, &tmp, &cur)?;

    cur = gelu(&mut ctx0, &cur)?;
    cur = conv_1d_2s(&mut ctx0, &model.e_conv_2_w, &cur)?;

    tmp = repeat(&mut ctx0, &model.e_conv_2_b, &cur)?;
    cur = add(&mut ctx0, &tmp, &cur)?;
    cur = gelu(&mut ctx0, &cur)?;

    let iter = 0;

    let e_pe_stride = model.e_pe.dim1();
    let e_pe_offset = model.e_pe.dim1() * model.e_pe.dtype_size() * n_ctx * iter;

    let e_pe = view_2d(
        &model.e_pe,
        model.e_pe.dim1(),
        n_ctx,
        e_pe_stride,
        e_pe_offset,
    )?;

    let mut inpL = add(&mut ctx0, &e_pe, &cur.transpose(0, 1)?)?;
    //let inp_L = &cur;
    for i1 in 0..n_layer {
        let mut ctx_l: TensorContext = TensorContext::new(&wctx.buf_compute_layer);
        let layer = model.layers_encoder.get(i1).unwrap();
        // norm
        {
            cur = norm(&mut ctx_l, &inpL)?;
            tmp = repeat(&mut ctx_l, &layer.attn_ln_0_w, &cur)?;
            cur = mul(&mut ctx_l, &tmp, &cur)?;
            tmp = repeat(&mut ctx_l, &layer.attn_ln_0_b, &cur)?;
            cur = add(&mut ctx_l, &tmp, &cur)?;
        }

        // self-attention
        {
            let mut qcur = matmul(&mut ctx_l, &layer.attn_q_w, &cur)?;
            tmp = repeat(&mut ctx_l, &layer.attn_q_b, &qcur)?;
            qcur = add(&mut ctx_l, &tmp, &qcur)?;
            let Kcur = matmul(&mut ctx_l, &layer.attn_k_w, &cur)?;
            let mut vcur = matmul(&mut ctx_l, &layer.attn_v_w, &cur)?;
            tmp = repeat(&mut ctx_l, &layer.attn_v_b, &vcur)?;
            vcur = add(&mut ctx_l, &tmp, &vcur)?;
            tmp = cpy(
                &qcur,
                &new_tensor_3d(&mut ctx_l, DType::F16, n_state / n_head, n_head, n_ctx)?,
            )?;

            //USE_FLASH_ATTN
            {
                let Q = tmp.permute(0, 2, 1, 3)?;

                tmp = cpy(
                    &Kcur,
                    &new_tensor_3d(&mut ctx_l, DType::F16, n_state / n_head, n_head, n_ctx)?,
                )?;

                let K = tmp.permute(0, 2, 1, 3)?;

                tmp = reshape_3d(&vcur, n_state / n_head, n_head, n_ctx)?;
                tmp = tmp.permute(1, 2, 0, 3)?;

                let V = cpy(
                    &tmp,
                    &new_tensor_3d(&mut ctx_l, DType::F16, n_ctx, n_state / n_head, n_head)?,
                )?;

                let mut KQV = flash_attn(&mut ctx_l, &Q, &K, &V)?;

                let KQV_merged = KQV.permute(0, 2, 1, 3)?;

                cur = cpy(
                    &KQV_merged,
                    &new_tensor_2d(&mut ctx_l, DType::F32, n_state, n_ctx)?,
                )?;
            }
        }

        // projection

        {
            cur = matmul(&mut ctx_l, &layer.attn_ln_1_w, &cur)?;
            tmp = repeat(&mut ctx_l, &layer.attn_ln_1_b, &cur)?;
            cur = add(&mut ctx_l, &tmp, &cur)?;
        }

        // add the input
        let inpFF = add(&mut ctx_l, &cur, &inpL)?;

        // feed-forward network
        {
            // norm
            {
                cur = norm(&mut ctx_l, &inpFF)?;
                tmp = repeat(&mut ctx_l, &layer.mlp_ln_w, &cur)?;
                tmp = mul(&mut ctx_l, &tmp, &cur)?;
                let tmp2 = repeat(&mut ctx_l, &layer.mlp_ln_b, &cur)?;
                cur = add(&mut ctx_l, &tmp, &tmp2)?;
                // cur = mlp_ln_w*cur + mlp_ln_b
            }
            // fully connected
            cur = matmul(&mut ctx_l, &layer.mlp_0_w, &cur)?;
            tmp = repeat(&mut ctx_l, &layer.mlp_0_b, &cur)?;
            cur = add(&mut ctx_l, &tmp, &cur)?;
            // GELU activation
            cur = gelu(&mut ctx_l, &cur)?;
            // projection
            cur = matmul(&mut ctx_l, &layer.mlp_1_w, &cur)?;

            tmp = repeat(&mut ctx_l, &layer.mlp_1_b, &cur)?;
            cur = add(&mut ctx_l, &tmp, &cur)?;
        }
        // output from this layer
        let inpO = add(&mut ctx_l, &cur, &inpFF)?;

        // layer end
        let nbytes = inpL.nbytes();
        inpL.as_bytes_mut()[..nbytes].copy_from_slice(&inpO.as_bytes()[..nbytes]);

        println!("end :{}", i1)
    }
    // layer end
    cur = inpL;
    // norm
    {
        cur = norm(&mut ctx0, &cur)?;
        tmp = repeat(&mut ctx0, &model.e_ln_w, &cur)?;
        tmp = mul(&mut ctx0, &tmp, &cur)?;
        let tmp2 = repeat(&mut ctx0, &model.e_ln_b, &cur)?;
        cur = add(&mut ctx0, &tmp, &tmp2)?;
        // cur = mlp_ln_w*cur + mlp_ln_b
    }

    // pre-compute cross-attention memory

    for il in 0..model.hparams.n_text_layer as usize {
        let layer = &model.layers_decoder[il];
        let mut Kcross = matmul(&mut ctx0, &layer.cross_attn_k_w, &cur)?;

        let tensor_f32 = new_f32_tensor(&mut ctx0, (n_state as f32 / n_head as f32).powf(-0.25))?;

        Kcross = scale(&mut ctx0, &Kcross, &tensor_f32)?;

        // let x: &[f32] = unsafe { Kcross.as_slice::<f32>() };
        // let mut sum: f64 = 0.0;
        // for i in 0..Kcross.elem_count() {
        //     sum += x[i].abs() as f64;
        // }

        // println!(
        //     "Kcross,sum:{:?},sha
        //     pe:{:?},stride:{:?}",
        //     sum,
        //     Kcross.ggml_shape(),
        //     Kcross.dim().stride_4d()
        // );

        // break;
        let mut Vcross = matmul(&mut ctx0, &layer.cross_attn_v_w, &cur)?;

        tmp = repeat(&mut ctx0, &layer.cross_attn_v_b, &Vcross)?;
        Vcross = add(&mut ctx0, &tmp, &Vcross)?;

        let k = view_1d(
            &model.memory_cross_k,
            n_state * n_ctx,
            (model.memory_cross_k.dtype_size() * n_state) * (il * n_ctx),
        )?;
        let v = view_1d(
            &model.memory_cross_v,
            n_state * n_ctx,
            (model.memory_cross_v.dtype_size() * n_state) * (il * n_ctx),
        )?;

        let k1 = cpy(&Kcross, &k)?;
        let v1 = cpy(&Vcross, &v)?;

        // let x: &[f32] = unsafe { k.as_slice::<f32>() };
        // let mut sum: f64 = 0.0;
        // for i in 0..k.elem_count() {
        //     sum += x[i].abs() as f64;
        // }

        // println!(
        //     "k1,sum:{:?},sha
        //     pe:{:?},stride:{:?}",
        //     sum,
        //     k1.ggml_shape(),
        //     k1.dim().stride_4d()
        // );

        // let x: &[f32] = unsafe { v.as_slice::<f32>() };
        // let mut sum: f64 = 0.0;
        // for i in 0..v.elem_count() {
        //     sum += x[i].abs() as f64;
        // }

        // println!(
        //     "v1,sum:{:?},sha
        //     pe:{:?},stride:{:?}",
        //     sum,
        //     v1.ggml_shape(),
        //     v1.dim().stride_4d()
        // );
        // break;
    }

    Ok(())
}

fn main() {
    let file_path = "/opt/cproject/whisper.cpp-1.0.3/samples/jfk.wav";
    let mut reader = hound::WavReader::open(file_path).unwrap();
    let s16: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
    println!("len:{}", s16.len());
    let samples = convert_integer_to_float_audio(&s16); //deinterleave_vecs_f32(&data.data(), data.channel_count() as usize);
    let model_path = "/opt/cproject/whisper.cpp-1.0.3/models/ggml-tiny.en.bin";
    let mut wctx = WhisperContext::new(model_path).unwrap();
    whisper_pcm_to_mel(&mut wctx, Arc::new(samples)).unwrap();
    whisper_encode(&mut wctx, 1, 0).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_model() {
        let file_path = "/opt/rsproject/chappie/jfk.wav";
        let mut reader = hound::WavReader::open(file_path).unwrap();
        let s16: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
        println!("len:{}", s16.len());
        let samples = convert_integer_to_float_audio(&s16); //deinterleave_vecs_f32(&data.data(), data.channel_count() as usize);
        let model_path = "/opt/cproject/whisper.cpp-1.0.3/models/ggml-tiny.en.bin";
        let mut wctx = WhisperContext::new(model_path).unwrap();
        whisper_pcm_to_mel(&mut wctx, Arc::new(samples)).unwrap();
    }

    #[derive(Debug)]
    struct A {
        a: i32,
        name: String,
    }

    struct B {
        a: Vec<A>,
    }

    #[test]
    fn test_a() {
        let mut x = A {
            a: 5,
            name: "123".to_string(),
        };
        let y = &mut x as *mut A;
        let b = B { a: vec![x] };

        if let Some(a1) = unsafe { y.as_ref() } {
            println!("{:?}", a1.name);
        }

        println!("{:?}", b.a)
    }
}
