import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)
import numpy as np
from copy import deepcopy
class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="./models/zrf/clip_vit_large_patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.token_embedding = self.transformer.text_model.embeddings.token_embedding.weight  # torch.Size([49408, 768])

        self.new_dis = torch.distributions.MultivariateNormal(torch.zeros(768).cuda(), torch.eye(768).cuda())
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, class_index):  # [ 3 个 'A high-quality image of the automobile'], 1
        if text[0] != '':  # 'A high-quality image of the automobile'
            if self.id_data == 'cifar100':
                fine_labels = [
                    'apples',  # id 0
                    'fish',
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
                    'chimp',
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
                    'keyboard',
                    'lamp',
                    'mower',
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
                    'truck',
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
                    'peppers',
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
                    'worm',
                ]
            elif self.id_data == 'in100':
                fine_labels = ['stingray', 'hen', 'magpie', 'kite', 'vulture',
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
            else:
                fine_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

            # 将输入的文本转换为 token ID 序列 tokens, automobile
            tmp_token = self.tokenizer([fine_labels[class_index]], truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")  # {'input_ids': tensor([[49406, 25258, 49407, 49407]]), 'length': tensor([3]), 'attention_mask': tensor([[1, 1, 1, 0, 0]])}
            tokens = tmp_token["input_ids"].to(self.device)  # tensor([[49406, 25258, 49407, 49407]], device='cuda:0')
            # token_embedding.weight 存储的是 CLIP 预训练的 token 级别的嵌入，包括常见的单词、子词和特殊标记。
            # token_embedding.weight: torch.Size([49408, 768]), tokens[0][1]: tensor(25258, device='cuda:0')
            # automobile 原始的 token 嵌入
            original_embed = deepcopy(self.transformer.text_model.embeddings.token_embedding.weight[tokens[0][1]])  # torch.Size([768])
            # automobile 的 token ID
            original_id = tokens[0][1]  # tensor(25258, device='cuda:0')
            print(self.transformer.text_model.embeddings.token_embedding.weight[original_id][:10])
            if False:
                noise = 0.03 * self.new_dis.rsample(
                    (1,)).squeeze()
                self.transformer.text_model.embeddings.token_embedding.weight[original_id] = noise + original_embed
            else:
                # 从 class_index 类的离群值中, 随机选择一个离群值
                # torch.Size([10, 10000, 768])[1] -> torch.Size([10000, 768])
                outlier = self.outlier_embedding[class_index][np.random.choice(10000, 1)[0]]  # torch.Size([768])
                # 用离群值替换 automobile 原始的文本 token 嵌入
                self.transformer.text_model.embeddings.token_embedding.weight[original_id] = outlier.cuda()

        # ['A high-quality image of the automobile', '...', 'A high-quality image of the automobile']
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")

        #                  A   high   -   quality image  of   the automobile
        # tensor([[49406, 320, 1400, 268,  3027,  2867, 539, 518,  25258, 49407],
        #         [49406, 320, 1400, 268,  3027,  2867, 539, 518,  25258, 49407],
        #         [49406, 320, 1400, 268,  3027,  2867, 539, 518,  25258, 49407]], device='cuda:0')
        tokens = batch_encoding["input_ids"].to(self.device)  # torch.Size([3, 77])

        # Stable Diffusion 生成的文本条件并不直接使用 token_embedding.weight 中的嵌入，
        # 而是通过 outputs = self.transformer(input_ids=tokens) 计算得到的 outputs.last_hidden_state 作为最终的文本嵌入。
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state  # torch.Size([3, 77, 768])
        # return to the intial embeddings for sampling next time.
        if text[0] != '':  # 'A high-quality image of the automobile'
            # 将 automobile 的 token 嵌入恢复为初始状态
            self.transformer.text_model.embeddings.token_embedding.weight[original_id] = original_embed
        return z  # torch.Size([3, 77, 768])

    # [ 3 个 'A high-quality image of the automobile'], 1, opt
    def encode(self, text, class_index, opt):
        self.id_data = opt.id_data  # cifar10
        # ./cifar10_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy
        self.outlier_embedding = torch.from_numpy(
                np.load(opt.loaded_embedding))  # torch.Size([10, 10000, 768])
        return self(text, class_index)  # [ 3 个 'A high-quality image of the automobile'], 1


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)

    # 生成 CIFAR-10 的文本 Anchor
    CIFAR10 = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    version = "./models/zrf/clip_vit_large_patch14"

    tokenizer = CLIPTokenizer.from_pretrained(version)
    transformer = CLIPTextModel.from_pretrained(version)
    token_embedding = transformer.text_model.embeddings.token_embedding.weight  # torch.Size([49408, 768])

    anchor_list = []

    for index in range(len(CIFAR10)):
        tmp_token = tokenizer([CIFAR10[index]], truncation=True, max_length=77,
                                   return_length=True,
                                   return_overflowing_tokens=False, padding="max_length", return_tensors="pt")  # {'input_ids': tensor([[49406, 16451, 49407, ... , 49407]]), 'length': tensor([3]), 'attention_mask': tensor([[1, 1, 1, 0, ... , 0]])}
        tokens = tmp_token["input_ids"].to("cuda")  # torch.Size([1, 77])  tensor([[49406, 16451, 49407, ... , 49407]], device='cuda:0')
        print(CIFAR10[index], tokens)
        original_embed = transformer.text_model.embeddings.token_embedding.weight[tokens[0][1]][None, ...]  # torch.Size([1, 768]) tokens[0][1]: tensor(16451, device='cuda:0')
        print("original_embed: ", original_embed.shape)
        anchor_list.append(original_embed)

    anchor = torch.cat(anchor_list, dim=0).detach().cpu().numpy()
    print(anchor.shape, type(anchor))

    # 保存为 token_embed_c10.npy 文件
    np.save("./token_embed_c10.npy", anchor)