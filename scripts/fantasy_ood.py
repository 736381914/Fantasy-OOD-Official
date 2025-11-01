import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
# 加载一个 Stable Diffusion 的“安全性检测模型”（safety checker）,
# 用来检测生成图像是否包含潜在的不安全内容（如 NSFW 图像），通常用于内容过滤。
safety_model_id = "./models/zrf/stable_diffusion_safety_checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def get_class_names(opt):
    if opt.id_data == 'in100':
        return ['stingray', 'hen', 'magpie', 'kite', 'vulture',
               'agama',   'tick', 'quail', 'hummingbird', 'koala',
               'jellyfish', 'snail', 'crawfish', 'flamingo', 'orca',
               'chihuahua', 'coyote', 'tabby', 'leopard', 'lion',
               'tiger','ladybug', 'fly' , 'ant', 'grasshopper',
               'monarch', 'starfish', 'hare', 'hamster', 'beaver',
               'zebra', 'pig', 'ox', 'impala',  'mink',
               'otter', 'gorilla', 'panda', 'sturgeon', 'accordion',
               'carrier', 'ambulance', 'apron', 'backpack', 'balloon',
               'banjo','barn','baseball', 'basketball', 'beacon',
               'binder', 'broom', 'candle', 'castle', 'chain',
               'chest', 'church', 'cinema', 'cradle', 'dam',
               'desk', 'dome', 'drum','envelope', 'forklift',
               'fountain', 'gown', 'hammer','jean', 'jeep',
               'knot', 'laptop', 'mower', 'library','lipstick',
               'mask', 'maze', 'microphone','microwave','missile',
                'nail', 'perfume','pillow','printer','purse',
               'rifle', 'sandal', 'screw','stage','stove',
               'swing','television','tractor','tripod','umbrella',
                'violin','whistle','wreck', 'broccoli', 'strawberry'
               ]
    elif opt.id_data == 'cifar100':
        return [
    'apples',  # id 0
    'aquarium fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottles',
    'bowls',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'cans',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cups',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'computer keyboard',
    'lamp',
    'lawn-mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple',
    'motorcycle',
    'mountain',
    'mouse',
    'mushrooms',
    'oak',
    'oranges',
    'orchids',
    'otter',
    'palm',
    'pears',
    'pickup truck',
    'pine',
    'plain',
    'plates',
    'poppies',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'roses',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflowers',
    'sweet peppers',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulips',
    'turtle',
    'wardrobe',
    'whale',
    'willow',
    'wolf',
    'woman',
    'worm']
    else:
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']



def get_prompt(opt):
    import random
    # 从 ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 随机选择一个
    chozen_class = random.choice(get_class_names(opt))  # automobile
    if chozen_class[0] in ['a', 'e', 'i', 'o', 'u']:  # a
        return 'A high-quality image of the ' + chozen_class, chozen_class  # ('A high-quality image of the automobile', 'automobile')
    else:
        return 'A high-quality image of the ' + chozen_class, chozen_class

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="/nobackup-fast/txt2img-samples-in100-demo/"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--id_data",
       type=str,
       default='in100'
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )

    parser.add_argument(
        "--gaussian_scale",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )

    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--loaded_embedding",
        type=str,
        default='/nobackup-slow/dataset/my_xfdu/diffusion/outlier_npos_embed.npy'
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/nobackup-slow/dataset/my_xfdu/diffusion/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    if opt.laion400m:  # False
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)  # 设置 Python、NumPy、PyTorch 的随机种子 42

    # 根据 YAML 构建模型结构
    config = OmegaConf.load(f"{opt.config}")  # configs/stable-diffusion/v1-inference.yaml
    # 从 .ckpt 文件中加载模型权重到这个结构上
    model = load_model_from_config(config, f"{opt.ckpt}")  # ./snapshots/sd-v1-4.ckpt

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # 扩散模型在生成图像时有两个阶段：
    # 1、前向扩散（Forward Diffusion）：给图像逐渐加噪 → 得到纯噪声
    # 2、反向采样（Reverse Sampling）：从噪声一步步去噪 → 恢复出图像
    # 而“采样器”就是控制 反向采样过程的算法，决定了：
    # 一共需要多少步（step）
    # 每一步如何预测下一张图
    # 图像质量、速度、稳定性等表现
    if opt.plms:
        # 创建 PLMS 采样器, 接收扩散模型 model，然后基于该模型从随机噪声一步步“反扩散”生成清晰图像。
        sampler = PLMSSampler(model)  # 比 DDIM 更快、更稳定, 生成质量也较高
    else:
        sampler = DDIMSampler(model)  # 更可控, 能生成较连贯图像

    os.makedirs(opt.outdir, exist_ok=True)  # ./snapshots/txt2img-samples-cifar10-demo/, 如果目标目录已经存在, 也不会报错
    outpath = opt.outdir

    # 给生成的图像添加一个“数字水印”（invisible watermark）的准备过程，常用于 Stable Diffusion 等模型中，标注图像的来源或模型信息
    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"  # 设置水印内容
    wm_encoder = WatermarkEncoder()  # 创建一个水印编码器对象，用于把字符串转换成一个可以嵌入图像的水印信号（一般是隐形的，不影响图像内容）
    # 把水印字符串(wm)转换成 UTF-8 编码的字节流(bytes), 然后设置进编码器。
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))  # 'bytes' 是告诉 set_watermark() 传入的是字节类型的水印。

    batch_size = opt.n_samples  # 3
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size  # 3
    if not opt.from_file:
        prompt = opt.prompt  # a painting of a virus monster playing guitar
        assert prompt is not None
        data = [batch_size * [prompt]]  # [['a painting of a virus monster playing guitar', 'a painting of a virus monster playing guitar', 'a painting of a virus monster playing guitar']]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    # ./snapshots/txt2img-samples-cifar10-demo/  samples
    sample_path = os.path.join(outpath, "samples")  # ./snapshots/txt2img-samples-cifar10-demo/samples
    os.makedirs(sample_path, exist_ok=True)  # ./snapshots/txt2img-samples-cifar10-demo/samples, 如果目标目录已经存在, 也不会报错
    base_count = len(os.listdir(sample_path)) + 2000000 * opt.index  # 0
    grid_count = len(os.listdir(outpath)) - 1  # 0

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    # autocast（自动混合精度） 主要用于 加速推理和训练
    # 自动将合适的操作转换为 半精度（float16），减少显存占用，提高计算速度
    # 关键部分仍保持 全精度（float32），避免精度损失导致的不稳定
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                # trange(n) 是 tqdm(range(n)) 的简写
                for n in trange(opt.n_iter, desc="Sampling"):  # 用于循环执行 653 次，同时显示进度条，用于跟踪任务进度。
                    for prompts in tqdm(data, desc="data"):  # ['a painting of a virus monster playing guitar', ... , 'a painting of a virus monster playing guitar']
                        uc = None
                        if opt.scale != 1.0:  # 7.5
                            # ['', '', ''], 0, opt
                            uc = model.get_learned_conditioning(batch_size * [""], 0, opt)  # torch.Size([3, 77, 768])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        prompts, chozen_class = get_prompt(opt)  # 'A high-quality image of the automobile', 'automobile'

                        # special cases, we need more prompts for help, otherwise the generated images will not look like ImageNet classes.
                        if chozen_class == 'kite' or chozen_class == 'quail':
                            prompts += ' bird'
                        if chozen_class == 'chest':
                            prompts += ' box'
                        if chozen_class == 'tick':
                            prompts += ' bite'
                        if chozen_class == 'stingray':
                            prompts += ' in the water'
                        if chozen_class == 'ox' or chozen_class == 'impala':
                            prompts += ' animal'
                        if chozen_class == 'nail':
                            prompts += 'A high-quality image of the wire nail'

                        # ['A high-quality image of the automobile'] * 3
                        prompts = [prompts] * opt.n_samples  # ['A high-quality image of the automobile', 'A high-quality image of the automobile', 'A high-quality image of the automobile']

                        # ['A high-quality image of the automobile' * 3], 1, opt
                        c = model.get_learned_conditioning(prompts, get_class_names(opt).index(chozen_class), opt)  # torch.Size([3, 77, 768])
                        # 0 * torch.Size([3, 77, 768])
                        c += opt.gaussian_scale * torch.randn(c.size(0), c.size(1), c.size(2)).cuda()  # torch.Size([3, 77, 768])

                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]  # [4, 64, 64]
                        # torch.Size([3, 4, 64, 64])
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,  # 扩散步数(越大, 生成质量越高, 但速度变慢), 50
                                                         conditioning=c,  # 文本条件，用于控制生成, torch.Size([3, 77, 768])
                                                         batch_size=opt.n_samples,  # 一次生成的图像数量, 3
                                                         shape=shape,  # 初始噪声 x_T 的形状, 并且会在采样过程中逐步去噪, 最终变成清晰的图像, 即 x_T 被初始化为 shape=[4, 64, 64] 的高斯噪声
                                                         verbose=False,  # 	是否打印调试信息
                                                         unconditional_guidance_scale=opt.scale,  # 引导因子(CFG Scale), 7.5, 用于控制文本对生成图像的影响
                                                         unconditional_conditioning=uc,  # 无条件嵌入(用于 Classifier-Free Guidance), torch.Size([3, 77, 768])
                                                         eta=opt.ddim_eta,  # 调整采样的随机性, 0.0, 表示最确定性的 DDIM 采样
                                                         x_T=start_code)  # 扩散过程的起始噪声（初始随机潜变量）

                        # 将潜空间(latent space)的表示转换回图像数据。
                        x_samples_ddim = model.decode_first_stage(samples_ddim)  # torch.Size([3, 3, 512, 512])
                        # 对 x_samples_ddim 进行 归一化 并 限制取值范围在 [0, 1] 之间
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)  # torch.Size([3, 3, 512, 512])
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()  # (3, 512, 512, 3)

                        # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)  # torch.Size([3, 3, 512, 512])
                        # 对 x_checked_image_torch 进行双线性插值, 将其调整为 256x256 的分辨率
                        x_checked_image_torch = torch.nn.functional.interpolate(x_checked_image_torch.float(), [256,256],
                                                                                mode='bilinear')  # torch.Size([3, 3, 256, 256])
                        if not opt.skip_save:  # True
                            # x_checked_image_torch: torch.Size([3, 3, 256, 256])
                            for x_sample in x_checked_image_torch:  # torch.Size([3, 256, 256])
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')  # (256, 256, 3)
                                img = Image.fromarray(x_sample.astype(np.uint8))  # <class 'PIL.Image.Image'>
                                # img = put_watermark(img, wm_encoder)
                                # './snapshots/txt2img-samples-cifar10-demo/samples/1', 如果目标目录已经存在, 也不会报错
                                os.makedirs(sample_path + '/' + str(get_class_names(opt).index(chozen_class)), exist_ok=True)
                                # './snapshots/txt2img-samples-cifar10-demo/samples/1/automobile_00000.png'
                                img.save(os.path.join(sample_path + '/' + str(get_class_names(opt).index(chozen_class)),
                                                      chozen_class + '_' + f"{base_count:05}.png"))
                                # img.save(os.path.join(sample_path, chozen_class + '_' + f"{base_count:05}.png"))
                                base_count += 1

                        if not opt.skip_grid:  # False
                            all_samples.append(x_checked_image_torch)

                if not opt.skip_grid:  # False
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    # img = put_watermark(img, wm_encoder)

                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()