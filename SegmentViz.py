from segment import Segmentation
from datasets.dataset import Dataset
from argparse import ArgumentParser
from tqdm import tqdm
import cv2
import json
from pathlib import Path

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
parser.add_argument("--save_as_baseline", action='store_true')
parser.add_argument("--no_load_hparams", action='store_true')

args = parser.parse_args()

if __name__ == '__main__':
    output_dir = Path('output') / Path(args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_load_hparams:
        parms_config = Path('search_results') / Path(args.dataset) / 'best_std_params.json'
        if not parms_config.exists():
            raise FileNotFoundError(f"Configuration file not found: {parms_config}")
        with open(parms_config, 'r') as f:
            best_params = json.load(f)
            args.alpha = best_params['params']['alpha']
            args.lam = best_params['params']['lam']
            args.eps = best_params['params']['eps']
            args.nb_iter = best_params['params']['nb_iter']

    seg = Segmentation(process=args.process, bs=args.bs, epochs=args.epochs, resolution=tuple(args.resolution), device=args.device,
                       activation=args.activation, loss_type=args.loss_type, threshold=args.threshold, conv_type=args.conv_type,
                       alpha=args.alpha, lam=args.lam, eps=args.eps, nb_iter=args.nb_iter, gen_init=args.gen_init)
    ds = Dataset(args.dataset)

    total_iou = 0
    total_samples = 0
    
    total_files = ds.loader

    with tqdm(ds.load_samples(), total=total_files, desc="Processing images") as pbar:
        for img, mask in pbar:
            cnt = pbar.n + 1
            try: 
                iou, seg_mask, _ = seg.segment(img, mask)
                total_iou += iou
                total_samples += 1
                
                current_miou = total_iou / total_samples
                pbar.set_postfix({
                    'Current IoU': f'{iou:.2f}',
                    'mIoU so far': f'{current_miou:.2f}'
                })

                # seg_mask(h, w) mask(h, w, 1) img(h, w, 3)
                h, w = seg_mask.shape
                if args.save_as_baseline:
                    cv2.imwrite(str(output_dir / f'{cnt:04d}_baseline.png'), seg_mask.reshape(h, w, 1) * 255)
                else:
                    cv2.imwrite(str(output_dir / f'{cnt:04d}_pred.png'), seg_mask.reshape(h, w, 1) * 255)
                cv2.imwrite(str(output_dir / f'{cnt:04d}_gt.png'), mask * 255)
                cv2.imwrite(str(output_dir / f'{cnt:04d}_img.png'), img)

            except Exception as e:
                print(e)
                continue
    
    final_miou = total_iou / total_samples
    print(f'Final mIoU: {final_miou:.4f}')
