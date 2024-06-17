use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use core::cell::UnsafeCell;
use galois::Shape;
use galois::Tensor as GsTensor;
use galois::{DType, GS_TYPE_SIZE};
use lazy_static::lazy_static;
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

// type GGMLFp16T = u16;

// #[derive(Clone, Copy)]
// #[repr(usize)]
// enum DType {
//     GGML_TYPE_I8,
//     GGML_TYPE_I16,
//     GGML_TYPE_I32,
//     F16,
//     F32,
//     GGML_TYPE_COUNT,
// }

// const get_type_size: [usize; DType::GGML_TYPE_COUNT as usize] = [
//     std::mem::size_of::<i8>(),
//     std::mem::size_of::<i16>(),
//     std::mem::size_of::<i32>(),
//     std::mem::size_of::<GGMLFp16T>(),
//     std::mem::size_of::<f32>(),
// ];

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
    #[error("Unexpected io: {0}")]
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

fn new_tensor_1d(
    ctx: &mut TensorContext,
    buf: &[u8],
    dtype: DType,
    ne0: usize,
) -> WsResult<GsTensor> {
    let dim = [ne0];
    new_tensor(ctx, buf, dtype, Shape::from_array(dim))
}

fn new_tensor_2d(
    ctx: &mut TensorContext,
    buf: &[u8],
    dtype: DType,
    ne0: usize,
    ne1: usize,
) -> WsResult<GsTensor> {
    let dim = [ne0, ne1];
    new_tensor(ctx, buf, dtype, Shape::from_array(dim))
}

fn new_tensor_3d(
    ctx: &mut TensorContext,
    buf: &[u8],
    dtype: DType,
    ne0: usize,
    ne1: usize,
    ne2: usize,
) -> WsResult<GsTensor> {
    let dim = [ne0, ne1, ne2];
    new_tensor(ctx, buf, dtype, Shape::from_array(dim))
}

fn new_tensor_4d(
    ctx: &mut TensorContext,
    buf: &[u8],
    dtype: DType,
    ne0: usize,
    ne1: usize,
    ne3: usize,
    ne4: usize,
) -> WsResult<GsTensor> {
    let dim = [ne0, ne1];
    new_tensor(ctx, buf, dtype, Shape::from_array(dim))
}

fn new_tensor(
    ctx: &mut TensorContext,
    buf: &[u8],
    dtype: DType,
    shape: Shape,
) -> WsResult<GsTensor> {
    let cur_offset = ctx.offset;
    let cur_size = ctx.size;
    let size_needed: usize = get_type_size(dtype) * shape.size();
    if cur_offset + size_needed > buf.len() {
        return Err(WsError::NotEnoughSpace);
    }
    let t = GsTensor::from_bytes(&buf[cur_offset..cur_offset + size_needed], shape, dtype);
    ctx.offset = cur_offset + size_needed;
    ctx.size = size_needed;
    ctx.n_objects += 1;
    Ok(t)
}

#[derive(Default)]
struct TensorContext {
    offset: usize,
    size: usize,
    n_objects: usize,
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
        let mut tensors: HashMap<String, *const GsTensor> = HashMap::new();

        let mut whisper_mode = {
            let mut tensor_ctx = TensorContext::default();
            let n_vocab = hparams.n_vocab as usize;
            let n_audio_ctx = hparams.n_audio_ctx as usize;
            let n_audio_state = hparams.n_audio_state as usize;
            let n_audio_layer = hparams.n_audio_layer as usize;

            let n_text_ctx = hparams.n_text_ctx as usize;
            let n_text_state = hparams.n_text_state as usize;
            let n_text_layer = hparams.n_text_layer as usize;
            let n_mels = hparams.n_mels as usize;

            // encoder
            let e_pe = new_tensor_2d(
                &mut tensor_ctx,
                buf_model,
                DType::F32,
                n_audio_state,
                n_audio_ctx,
            )?;
            let e_conv_1_w =
                new_tensor_3d(&mut tensor_ctx, buf_model, wtype, 3, n_mels, n_audio_state)?;
            let e_conv_1_b =
                new_tensor_2d(&mut tensor_ctx, buf_model, DType::F32, 1, n_audio_state)?;

            let e_conv_2_w = new_tensor_3d(
                &mut tensor_ctx,
                buf_model,
                wtype,
                3,
                n_audio_state,
                n_audio_state,
            )?;
            let e_conv_2_b =
                new_tensor_2d(&mut tensor_ctx, buf_model, DType::F32, 1, n_audio_state)?;

            let e_ln_w = new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_audio_state)?;
            let e_ln_b = new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_audio_state)?;

            // let weights = Box::new([
            //     e_pe, e_conv_1_w, e_conv_1_b, e_conv_2_w, e_conv_2_b, e_ln_w, e_ln_b, d_pe, d_te,
            //     d_ln_w, d_ln_b,
            // ]);
            tensors.insert(
                "encoder.positional_embedding".to_string(),
                &e_pe as *const GsTensor,
            );
            tensors.insert(
                "encoder.conv1.weight".to_string(),
                &e_conv_1_w as *const GsTensor,
            );
            tensors.insert(
                "encoder.conv1.bias".to_string(),
                &e_conv_1_b as *const GsTensor,
            );
            tensors.insert(
                "encoder.conv2.weight".to_string(),
                &e_conv_2_w as *const GsTensor,
            );
            tensors.insert(
                "encoder.conv2.bias".to_string(),
                &e_conv_2_b as *const GsTensor,
            );
            tensors.insert(
                "encoder.ln_post.weight".to_string(),
                &e_ln_w as *const GsTensor,
            );
            tensors.insert(
                "encoder.ln_post.bias".to_string(),
                &e_ln_b as *const GsTensor,
            );

            let mut layers_encoder: Vec<WhisperLayerEncoder> = Vec::with_capacity(n_audio_layer);
            let mut layers_decoder: Vec<WhisperLayerDecoder> = Vec::with_capacity(n_text_layer);
            for i in 0..n_audio_layer {
                let mlp_ln_w =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_audio_state)?;
                let mlp_ln_b =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_audio_state)?;
                let mlp_0_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    n_audio_state,
                    4 * n_audio_state,
                )?;
                let mlp_0_b =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, 4 * n_audio_state)?;
                let mlp_1_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    4 * n_audio_state,
                    n_audio_state,
                )?;
                let mlp_1_b = new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_audio_state)?;
                let attn_ln_0_w =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_audio_state)?;
                let attn_ln_0_b =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_audio_state)?;
                let attn_q_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    n_audio_state,
                    n_audio_state,
                )?;
                let attn_q_b =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_audio_state)?;

                let attn_k_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    n_audio_state,
                    n_audio_state,
                )?;

                let attn_v_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    n_audio_state,
                    n_audio_state,
                )?;
                let attn_v_b =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_audio_state)?;

                let attn_ln_1_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    n_audio_state,
                    n_audio_state,
                )?;
                let attn_ln_1_b =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_audio_state)?;

                tensors.insert(
                    format!("encoder.blocks.{}.mlp_ln.weight", i),
                    &mlp_ln_w as *const GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.mlp_ln.bias", i),
                    &mlp_ln_b as *const GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.mlp.0.weight", i),
                    &mlp_0_w as *const GsTensor,
                );
                tensors.insert(
                    format!("encoder.blocks.{}.mlp.0.bias", i),
                    &mlp_0_b as *const GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.mlp.2.weight", i),
                    &mlp_1_w as *const GsTensor,
                );
                tensors.insert(
                    format!("encoder.blocks.{}.mlp.2.bias", i),
                    &mlp_1_b as *const GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn_ln.weight", i),
                    &mlp_1_b as *const GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn_ln.bias", i),
                    &attn_ln_0_b as *const GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn.query.weight", i),
                    &attn_q_w as *const GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn.query.bias", i),
                    &attn_q_b as *const GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn.key.weight", i),
                    &attn_k_w as *const GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn.value.weight", i),
                    &attn_v_w as *const GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn.value.bias", i),
                    &attn_v_b as *const GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn.out.weight", i),
                    &attn_ln_1_w as *const GsTensor,
                );

                tensors.insert(
                    format!("encoder.blocks.{}.attn.out.bias", i),
                    &attn_ln_1_b as *const GsTensor,
                );

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
                })
            }

            let d_pe = new_tensor_2d(
                &mut tensor_ctx,
                buf_model,
                DType::F32,
                n_text_state,
                n_text_ctx,
            )?;

            let d_te = new_tensor_2d(&mut tensor_ctx, buf_model, wtype, n_text_state, n_vocab)?;
            let d_ln_w = new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;
            let d_ln_b = new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;

            tensors.insert(
                "decoder.positional_embedding".to_string(),
                &d_pe as *const GsTensor,
            );
            tensors.insert(
                "decoder.token_embedding.weight".to_string(),
                &d_te as *const GsTensor,
            );
            tensors.insert("decoder.ln.weight".to_string(), &d_ln_w as *const GsTensor);
            tensors.insert("decoder.ln.bias".to_string(), &d_ln_b as *const GsTensor);

            for i in 0..n_text_layer {
                let mlp_ln_w = new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;
                let mlp_ln_b = new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;

                let mlp_0_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    n_text_state,
                    4 * n_text_state,
                )?;
                let mlp_0_b =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, 4 * n_text_state)?;

                let mlp_1_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    4 * n_text_state,
                    n_text_state,
                )?;
                let mlp_1_b = new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;

                let attn_ln_0_w =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;
                let attn_ln_0_b =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;

                let attn_q_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    n_text_state,
                    n_text_state,
                )?;
                let attn_q_b = new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;

                let attn_k_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    n_text_state,
                    n_text_state,
                )?;

                let attn_v_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    n_text_state,
                    n_text_state,
                )?;
                let attn_v_b = new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;

                let attn_ln_1_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    n_text_state,
                    n_text_state,
                )?;
                let attn_ln_1_b =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;

                let cross_attn_ln_0_w =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;
                let cross_attn_ln_0_b =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;

                let cross_attn_q_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    n_text_state,
                    n_text_state,
                )?;
                let cross_attn_q_b =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;

                let cross_attn_k_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    n_text_state,
                    n_text_state,
                )?;

                let cross_attn_v_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    n_text_state,
                    n_text_state,
                )?;
                let cross_attn_v_b =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;

                let cross_attn_ln_1_w = new_tensor_2d(
                    &mut tensor_ctx,
                    buf_model,
                    wtype,
                    n_text_state,
                    n_text_state,
                )?;
                let cross_attn_ln_1_b =
                    new_tensor_1d(&mut tensor_ctx, buf_model, DType::F32, n_text_state)?;

                tensors.insert(
                    format!("decoder.blocks.{}.mlp_ln.weight", i),
                    &mlp_ln_w as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.mlp_ln.bias", i),
                    &mlp_ln_b as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.mlp.0.weight", i),
                    &mlp_0_w as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.mlp.0.bias", i),
                    &mlp_0_b as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.mlp.2.weight", i),
                    &mlp_1_w as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.mlp.2.bias", i),
                    &mlp_1_b as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn_ln.weight", i),
                    &attn_ln_0_w as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn_ln.bias", i),
                    &attn_ln_0_b as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn.query.weight", i),
                    &attn_q_w as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn.query.bias", i),
                    &attn_q_b as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn.key.weight", i),
                    &attn_k_w as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn.value.weight", i),
                    &attn_v_w as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn.value.bias", i),
                    &attn_v_b as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn.out.weight", i),
                    &attn_ln_1_w as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.attn.out.bias", i),
                    &attn_ln_1_b as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn_ln.weight", i),
                    &cross_attn_ln_0_w as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn_ln.bias", i),
                    &cross_attn_ln_0_b as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn.query.weight", i),
                    &cross_attn_q_w as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn.query.bias", i),
                    &cross_attn_q_b as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn.key.weight", i),
                    &cross_attn_k_w as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn.value.weight", i),
                    &cross_attn_v_w as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn.value.bias", i),
                    &cross_attn_v_b as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn.out.weight", i),
                    &cross_attn_ln_1_w as *const GsTensor,
                );
                tensors.insert(
                    format!("decoder.blocks.{}.cross_attn.out.bias", i),
                    &cross_attn_ln_1_b as *const GsTensor,
                );

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
                })
            }

            //load kv memmory
            let mut tensor_ctx: TensorContext = TensorContext::default();

            let n_text_ctx = hparams.n_text_ctx as usize;
            let n_text_state = hparams.n_text_state as usize;
            let n_text_layer = hparams.n_text_layer as usize;

            let n_mem = n_text_layer * n_text_ctx;
            let n_elements = n_text_state * n_mem;

            let memory_k = new_tensor_1d(&mut tensor_ctx, &buf_memory, DType::F16, n_elements)?;
            let memory_v = new_tensor_1d(&mut tensor_ctx, &buf_memory, DType::F16, n_elements)?;

            let n_audio_ctx = hparams.n_audio_ctx as usize;

            let n_mem = n_text_layer * n_audio_ctx;
            let n_elements = n_text_state * n_mem;
            let memory_cross_k =
                new_tensor_1d(&mut tensor_ctx, &buf_memory, DType::F16, n_elements)?;
            let memory_cross_v =
                new_tensor_1d(&mut tensor_ctx, &buf_memory, DType::F16, n_elements)?;

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
                for i in 0..n_dims as usize {
                    ne[i] = r.read_i32::<Endian>()? as usize;
                    nelements *= ne[i];
                }
                //  println!("nelements:{}", nelements);
                let mut buffer = vec![0; length as usize];
                r.read_exact(&mut buffer)?;
                let name = String::from_utf8_lossy(&buffer).to_string();
                let ref_tensor = tensors
                    .get(name.as_str())
                    .ok_or(WsError::UnknownTensor(name.clone()))?;

                if let Some(tensor) = unsafe { ref_tensor.as_ref() } {
                    if tensor.elem_count() != nelements {
                        return Err(WsError::WrongSizeTensor(
                            name,
                            tensor.elem_count(),
                            nelements,
                        ));
                    }
                    let shape = tensor.dim().shape().as_slice();
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
                    r.read_exact(tensor.as_bytes_mut())?;
                    //  println!("name:{}", name);
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

                    // println!("Referenced struct: {:?}", tensor.dim());
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
    let data = unsafe { ctx.mel.data.borrow() };
    let x: f32 = data.iter().sum();
    println!("x:{:?}", x);
    Ok(())
}

fn whisper_full(ctx: &mut WhisperContext, samples: Vec<f32>) {}

fn main() {
    let file_path = "/opt/cproject/whisper.cpp-1.0.3/samples/jfk.wav";
    let mut reader = hound::WavReader::open(file_path).unwrap();
    let s16: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
    println!("len:{}", s16.len());
    let samples = convert_integer_to_float_audio(&s16); //deinterleave_vecs_f32(&data.data(), data.channel_count() as usize);
    let model_path = "/opt/cproject/whisper.cpp-1.0.3/models/ggml-tiny.en.bin";
    let mut wctx = WhisperContext::new(model_path).unwrap();
    whisper_pcm_to_mel(&mut wctx, Arc::new(samples)).unwrap();
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
        a: A,
    }

    #[test]
    fn test_a() {
        let x = A {
            a: 5,
            name: "123".to_string(),
        };
        let y = &x as *const A;
        let b = B { a: x };

        if let Some(a1) = unsafe { y.as_ref() } {
            println!("{:?}", a1.name);
        }
        drop(y);
        println!("{:?}", b.a)
    }
}
