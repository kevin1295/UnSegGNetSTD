from segment import Segmentation
from datasets.dataset import Dataset
from argparse import ArgumentParser
from cli_utils import set_seed, str2bool

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
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default='model.pt')

args = parser.parse_args()

if __name__ == '__main__':
    set_seed(args.seed)
    seg = Segmentation(process=args.process, bs=args.bs, epochs=args.epochs, resolution=tuple(args.resolution), device=args.device,
                       activation=args.activation, loss_type=args.loss_type, threshold=args.threshold, conv_type=args.conv_type,
                       alpha=args.alpha, lam=args.lam, eps=args.eps, nb_iter=args.nb_iter,
                       gen_init=True, checkpoint_path=args.checkpoint)
    ds = Dataset(args.dataset)

    total_iou = 0
    total_samples = 0
    failed_samples = 0
    for sample_idx, (img, mask) in enumerate(ds.load_samples(), start=1):
        try:
            iou, _, _ = seg.segment(img, mask)
        except Exception as exc:
            failed_samples += 1
            print(f'Failed sample {sample_idx}: {exc}')
            continue

        total_iou += iou
        total_samples += 1
        print(f"IoU for Current Image: {iou:.2f}  mIoU so far: {(total_iou / total_samples):.2f}")

    if total_samples == 0:
        raise RuntimeError('No samples were evaluated successfully.')
    print(f'Final mIoU: {(total_iou / total_samples):.4f} '
          f'({total_samples} successful, {failed_samples} failed)')
