import random
import numpy as np
import torch
from segment import Segmentation
from datasets.dataset import Dataset
from argparse import ArgumentParser
from tqdm import tqdm

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def get_base_argparser():
    parser = ArgumentParser()
    parser.add_argument("--bs", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--resolution", nargs=2, type=int, default=[224, 224])
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--activation", type=str, default='selu')
    parser.add_argument("--loss_type", type=str, default='DMON')
    parser.add_argument("--process", type=str, default='DINO')
    parser.add_argument("--dataset", type=str, default='ECSSD')
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--conv_type", type=str, default='GCN')
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=3.0)
    parser.add_argument("--nb_iter", type=int, default=3)
    parser.add_argument("--gen_init", action='store_true')

    return parser

def train_fn(args, gen_init=False):
    seg = Segmentation(process=args.process, bs=args.bs, epochs=args.epochs, resolution=tuple(args.resolution), device=args.device,
                       activation=args.activation, loss_type=args.loss_type, threshold=args.threshold, conv_type=args.conv_type,
                       alpha=args.alpha, lam=args.lam, eps=args.eps, nb_iter=args.nb_iter, gen_init=args.gen_init)
    ds = Dataset(args.dataset)

    total_iou = 0
    total_samples = 0
    
    # 使用 ds.loader 作为总样本数
    total_files = ds.loader
    
    with tqdm(ds.load_samples(), total=total_files, desc="Processing images") as pbar:
        for img, mask in pbar:
            try: 
                iou, _, _ = seg.segment(img, mask)
                total_iou += iou
                total_samples += 1
                
                # 在进度条中显示当前 IoU 和累计 mIoU
                current_miou = total_iou / total_samples
                pbar.set_postfix({
                    'Current IoU': f'{iou:.2f}',
                    'mIoU so far': f'{current_miou:.2f}'
                })

            except Exception as e:
                print(e)
                continue
    
    final_miou = total_iou / total_samples
    print(f'Final mIoU: {final_miou:.4f}')
    return final_miou

if __name__ == '__main__':
    parser = get_base_argparser()
    args = parser.parse_args()
    train_fn(args)