import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_contour
from train_fn import get_base_argparser, train_fn
from pathlib import Path
import json


def objective(trial):
    # 获取基础参数解析器
    global args
    
    # 定义需要搜索的参数范围
    args.alpha = trial.suggest_float('alpha', 0.1, 10.0, log=True)
    args.lam = trial.suggest_float('lam', 0.1, 10.0, log=True)
    args.eps = trial.suggest_float('eps', 0.1, 10.0, log=True)
    args.nb_iter = trial.suggest_int('nb_iter', 1, 10)
    
    # 运行训练并返回 mIoU
    return train_fn(args)

if __name__ == '__main__':
    parser = get_base_argparser()
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--study_name', type=str)
    args = parser.parse_args()
    
    # 创建结果目录
    results_base_dir = Path('search_results')
    results_dir = results_base_dir / Path(args.dataset)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置 Optuna 存储 (SQLite)
    storage_url = f'sqlite:///{results_base_dir}/optuna.db'
    
    # 创建 Optuna 研究，使用存储后端
    study_name = args.study_name or f'std_param_search_{args.dataset}'
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )
    
    # 运行参数搜索，最多进行 50 次试验
    study.optimize(objective, n_trials=args.n_trials)
    
    # 输出最佳参数和结果
    print('Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value:.4f}')
    print('  Params:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
    
    # 保存最佳参数到 JSON 文件
    best_params = {
        'params': trial.params,
        'value': trial.value
    }
    
    with open(results_dir / 'best_std_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # 输出所有试验的结果
    print('\nAll trials:')
    for i, trial in enumerate(study.trials):
        print(f'Trial {i+1}: value={trial.value:.4f}, params={trial.params}')
    
    print('\nResults saved to:')
    print(f'- {results_base_dir / "optuna.db"} (SQLite database)')
    print(f'- {results_dir}/best_std_params.json (Best STD parameters)')
    
    # Optuna dashboard command
    # Generate and save visualizations
    vis_dir = results_dir / 'std_visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Optimization history
    fig = plot_optimization_history(study)
    fig.write_image(vis_dir / 'optimization_history.png')
    
    # Parameter importances
    fig = plot_param_importances(study)
    fig.write_image(vis_dir / 'param_importances.png')
    
    # Slice plot
    fig = plot_slice(study)
    fig.write_image(vis_dir / 'slice_plot.png')
    
    # Contour plots for parameter pairs
    param_pairs = [('alpha', 'lam'), ('alpha', 'eps'), ('lam', 'eps'), ('alpha', 'nb_iter'), ('lam', 'nb_iter'), ('eps', 'nb_iter')]
    for param1, param2 in param_pairs:
        try:
            fig = plot_contour(study, params=[param1, param2])
            fig.write_image(vis_dir / f'contour_{param1}_{param2}.png')
        except (RuntimeError, ValueError) as exc:
            print(f"Error plotting contour for {param1} and {param2}: {exc}")
    
    print('\nResults saved to:')
    print(f'- {results_dir}/ (SQLite database)')
    print(f'- {results_dir}/best_std_params.json (Best STD parameters)')
    print(f'- {vis_dir}/ (Visualization plots)')
    
    # Optuna dashboard command
    print('\nTo view the optimization results with Optuna Dashboard, run:')
    print(f'optuna dashboard --storage {storage_url}')
    
    # To load and analyze studies later, you can use:
    # study = optuna.load_study(study_name='std_param_search', storage=storage_url)
