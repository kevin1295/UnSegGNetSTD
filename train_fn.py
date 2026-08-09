from segment import Segmentation
from datasets.dataset import Dataset
from argparse import ArgumentParser
from tqdm import tqdm
from cli_utils import set_seed, str2bool


def get_base_argparser():
    parser = ArgumentParser()
    parser.add_argument("--bs", type=str2bool, default=False)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default='model.pt')

    return parser

def train_fn(args, gen_init=None):
    if gen_init is None:
        gen_init = args.gen_init
    set_seed(args.seed)
    seg = Segmentation(process=args.process, bs=args.bs, epochs=args.epochs, resolution=tuple(args.resolution), device=args.device,
                       activation=args.activation, loss_type=args.loss_type, threshold=args.threshold, conv_type=args.conv_type,
                       alpha=args.alpha, lam=args.lam, eps=args.eps, nb_iter=args.nb_iter,
                       gen_init=gen_init, checkpoint_path=args.checkpoint)
    ds = Dataset(args.dataset)

    total_iou = 0
    total_samples = 0
    
    # 使用 ds.loader 作为总样本数
    total_files = ds.loader
    
    failed_samples = 0
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

            except Exception as exc:
                failed_samples += 1
                print(f'Failed sample {total_samples + failed_samples}: {exc}')
                continue
    
    if total_samples == 0:
        raise RuntimeError('No samples were evaluated successfully.')
    final_miou = total_iou / total_samples
    print(f'Final mIoU: {final_miou:.4f} '
          f'({total_samples} successful, {failed_samples} failed)')
    return final_miou

if __name__ == '__main__':
    parser = get_base_argparser()
    args = parser.parse_args()
    train_fn(args)
