import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel


def get_anchor_cifar10():
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
                              return_overflowing_tokens=False, padding="max_length",
                              return_tensors="pt")  # {'input_ids': tensor([[49406, 16451, 49407, ... , 49407]]), 'length': tensor([3]), 'attention_mask': tensor([[1, 1, 1, 0, ... , 0]])}
        tokens = tmp_token["input_ids"].to(
            "cuda")  # torch.Size([1, 77])  tensor([[49406, 16451, 49407, ... , 49407]], device='cuda:0')
        print(CIFAR10[index], tokens)
        original_embed = transformer.text_model.embeddings.token_embedding.weight[tokens[0][1]][
            None, ...]  # torch.Size([1, 768]) tokens[0][1]: tensor(16451, device='cuda:0')
        print("original_embed: ", original_embed.shape)
        anchor_list.append(original_embed)

    anchor = torch.cat(anchor_list, dim=0).detach().cpu().numpy()
    print(anchor.shape, type(anchor))

    # 保存为 token_embed_c10.npy 文件
    np.save("./token_embed_c10.npy", anchor)

if __name__ == '__main__':
    get_anchor_cifar10()
