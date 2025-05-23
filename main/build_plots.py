import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_aggregated_results(mix, noise, all_avg_balances, all_tapes, output_dir):
    """Plot aggregated time-series results for a specific mix and noise level."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter data for this mix and noise
    avg_balance = all_avg_balances[(all_avg_balances['mix'] == mix) & (all_avg_balances['noise'] == noise)]
    tape = all_tapes[(all_tapes['mix'] == mix) & (all_tapes['noise'] == noise)]

    if avg_balance.empty or tape.empty:
        print(f"No data for mix={mix}, noise={noise}")
        return

    # Aggregate across trials
    avg_balance = avg_balance.drop(columns=['trial', 'mix', 'noise']).groupby('time').mean().reset_index()
    prop_traders = [col for col in avg_balance.columns if col.startswith('P')]

    # Plot 1: Average balance over time for prop traders
    plt.figure(figsize=(10, 6))
    for tid in prop_traders:
        plt.plot(avg_balance['time'], avg_balance[tid], label=f'Prop ({tid})')
    plt.title(f'Average Prop Trader Balance - {mix}, Noise {noise}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Balance')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{mix}_noise{noise}_balance.png'))
    plt.close()

    # Plot 2: Average bid-ask spread over time
    plt.figure(figsize=(10, 6))
    plt.plot(avg_balance['time'], avg_balance['spread'], label='Spread', color='red')
    plt.title(f'Average Bid-Ask Spread - {mix}, Noise {noise}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spread')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{mix}_noise{noise}_spread.png'))
    plt.close()

    # Plot 3: Trade prices (scatter all trades)
    plt.figure(figsize=(10, 6))
    plt.scatter(tape['time'], tape['price'], s=10, alpha=0.5)
    plt.title(f'Trade Prices - {mix}, Noise {noise}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Price')
    plt.savefig(os.path.join(output_dir, f'{mix}_noise{noise}_trade_prices.png'))
    plt.close()

def plot_summary_results(results_df, output_dir):
    """Plot summary results across all mixes and noise levels."""
    os.makedirs(output_dir, exist_ok=True)

    # Summary Plot 1: Trade frequency by noise and trader type (prop traders)
    plt.figure(figsize=(12, 8))
    prop_df = results_df[results_df['trader_id'].str.startswith('P')]
    sns.boxplot(x='noise', y='trades', hue='trader_type', data=prop_df)
    plt.title('Trade Frequency by Noise and Trader Type (Prop Traders)')
    plt.savefig(os.path.join(output_dir, 'summary_trade_frequency.png'))
    plt.close()

    # Summary Plot 2: Profit by noise and trader type (prop traders)
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='noise', y='profit', hue='trader_type', data=prop_df)
    plt.title('Profit by Noise and Trader Type (Prop Traders)')
    plt.savefig(os.path.join(output_dir, 'summary_profit.png'))
    plt.close()

def plot_baseline_results(base_path, output_dir):
    """Plot baseline results (retained from original code)."""
    all_avg_balances = []
    all_tapes = []
    all_results = []

    # Aggregate data across trials
    for trial_dir in os.listdir(base_path):
        if not trial_dir.startswith('trial'):
            continue
        trial_path = os.path.join(base_path, trial_dir)
        if not os.path.isdir(trial_path):
            continue
        trial_id = f"baseline_mix_1_noise0.00_{trial_dir}"
        avg_balance = pd.read_csv(os.path.join(trial_path, f'{trial_id}_avg_balance.csv'))
        blotters = pd.read_csv(os.path.join(trial_path, f'{trial_id}_blotters.csv'))
        tape = pd.read_csv(os.path.join(trial_path, f'{trial_id}_tape.csv'))

        avg_balance['trial'] = trial_id
        tape['trial'] = trial_id
        all_avg_balances.append(avg_balance)
        all_tapes.append(tape)

        prop_traders = [col for col in avg_balance.columns if col.startswith('P')]
        trader_types = {tid: 'SHVR' for tid in prop_traders}
        trade_counts = {}
        avg_balances = {}
        for tid in blotters['tid'].unique():
            trades = blotters[blotters['tid'] == tid]
            trade_counts[tid] = len(trades)
            if tid.startswith('B'):
                avg_balances[tid] = avg_balance['Buyer_ZIC_Avg'].iloc[-1]
            elif tid.startswith('S'):
                avg_balances[tid] = avg_balance['Seller_ZIC_Avg'].iloc[-1]
            elif tid.startswith('P'):
                initial_balance = 10000
                avg_balances[tid] = avg_balance[tid].iloc[-1] - initial_balance

        spread_avg = avg_balance['spread'].mean() if not avg_balance.empty else 0
        for tid in trade_counts.keys():
            noise = float(trial_id.split('noise')[1].split('_')[0])
            mix = 'baseline'
            trader_type = 'ZIC' if tid.startswith(('B', 'S')) else 'SHVR'
            all_results.append({
                'trial': trial_id,
                'noise': f'{noise:.2f}',
                'mix': mix,
                'trader_id': tid,
                'trader_type': trader_type,
                'profit': avg_balances[tid] if tid.startswith('P') else None,
                'volatility': avg_balance[tid].std() if tid in avg_balance.columns else None,
                'trades': trade_counts[tid],
                'price_std': tape['price'].std() if not tape.empty else 0,
                'spread_avg': spread_avg,
                'avg_balance': avg_balances[tid] if tid.startswith(('B', 'S')) else None
            })

    # Merge and average data
    all_avg_balances = pd.concat(all_avg_balances).drop(columns=['trial']).groupby('time').mean().reset_index()
    all_tapes = pd.concat(all_tapes)
    results_df = pd.DataFrame(all_results)

    # Plot baseline aggregated results
    output_dir = os.path.join(base_path, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Average balance over time for prop traders
    plt.figure(figsize=(10, 6))
    for tid in [col for col in all_avg_balances.columns if col.startswith('P')]:
        plt.plot(all_avg_balances['time'], all_avg_balances[tid], label=f'SHVR ({tid})')
    plt.title('Average Balance Over Time - Baseline')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Balance')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'avg_balance.png'))
    plt.close()

    # Plot 2: Trade prices (scatter all trades)
    plt.figure(figsize=(10, 6))
    plt.scatter(all_tapes['time'], all_tapes['price'], s=10)
    plt.title('Trade Prices - Baseline')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Price')
    plt.savefig(os.path.join(output_dir, 'trade_prices.png'))
    plt.close()

    # Plot 3: Average bid-ask spread over time
    plt.figure(figsize=(10, 6))
    plt.plot(all_avg_balances['time'], all_avg_balances['spread'], label='Spread', color='red')
    plt.title('Average Bid-Ask Spread Over Time - Baseline')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spread')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'avg_spread.png'))
    plt.close()

    return results_df

if __name__ == "__main__":
    # Configuration
    generate_baseline_plots = False  # Toggle for baseline plots
    output_base = 'output'
    experiment_base = os.path.join(output_base, 'experiment')
    baseline_base = os.path.join(output_base, 'baseline')

    all_avg_balances = []
    all_tapes = []
    all_results = []

    # Process Baseline (if enabled)
    baseline_results_df = None
    if generate_baseline_plots:
        print("Processing baseline...")
        baseline_results_df = plot_baseline_results(baseline_base, os.path.join(baseline_base, 'plots'))

    # Process Experiments
    print("Processing experiments...")
    for mix in ['mix_1', 'mix_2']:
        mix_path = os.path.join(experiment_base, mix)
        if not os.path.isdir(mix_path):
            continue
        for noise_dir in os.listdir(mix_path):
            if not noise_dir.startswith('noise_'):
                continue
            noise = noise_dir.replace('noise_', '')
            noise_path = os.path.join(mix_path, noise_dir)
            for trial_dir in os.listdir(noise_path):
                if not trial_dir.startswith('trial'):
                    continue
                trial_path = os.path.join(noise_path, trial_dir)
                if not os.path.isdir(trial_path):
                    continue
                trial_id = f"experiment_{mix}_noise{noise}_trial{trial_dir.replace('trial', '')}"
                avg_balance = pd.read_csv(os.path.join(trial_path, f'{trial_id}_avg_balance.csv'))
                blotters = pd.read_csv(os.path.join(trial_path, f'{trial_id}_blotters.csv'))
                tape = pd.read_csv(os.path.join(trial_path, f'{trial_id}_tape.csv'))

                avg_balance['trial'] = trial_id
                avg_balance['mix'] = mix
                avg_balance['noise'] = noise
                tape['trial'] = trial_id
                tape['mix'] = mix
                tape['noise'] = noise
                all_avg_balances.append(avg_balance)
                all_tapes.append(tape)

                prop_traders = [col for col in avg_balance.columns if col.startswith('P')]
                trade_counts = {}
                avg_balances = {}
                for tid in blotters['tid'].unique():
                    trades = blotters[blotters['tid'] == tid]
                    trade_counts[tid] = len(trades)
                    if tid.startswith('B'):
                        avg_balances[tid] = avg_balance['Buyer_ZIC_Avg'].iloc[-1]
                    elif tid.startswith('S'):
                        avg_balances[tid] = avg_balance['Seller_ZIC_Avg'].iloc[-1]
                    elif tid.startswith('P'):
                        initial_balance = 10000
                        avg_balances[tid] = avg_balance[tid].iloc[-1] - initial_balance

                spread_avg = avg_balance['spread'].mean() if not avg_balance.empty else 0
                for tid in trade_counts.keys():
                    trader_type = 'ZIC' if tid.startswith(('B', 'S')) else \
                                  'SHVR' if tid in ['P00', 'P01'] else \
                                  'TrendFollower' if 'TrendFollower' in blotters[blotters['tid'] == tid]['ttype'].values else \
                                  'MeanReverter' if 'MeanReverter' in blotters[blotters['tid'] == tid]['ttype'].values else \
                                  'RLAgent'
                    all_results.append({
                        'trial': trial_id,
                        'noise': noise,
                        'mix': mix,
                        'trader_id': tid,
                        'trader_type': trader_type,
                        'profit': avg_balances[tid] if tid.startswith('P') else None,
                        'volatility': avg_balance[tid].std() if tid in avg_balance.columns else None,
                        'trades': trade_counts[tid],
                        'price_std': tape['price'].std() if not tape.empty else 0,
                        'spread_avg': spread_avg,
                        'avg_balance': avg_balances[tid] if tid.startswith(('B', 'S')) else None
                    })

    # Concatenate and plot experiment results
    if all_avg_balances and all_tapes:
        all_avg_balances = pd.concat(all_avg_balances)
        all_tapes = pd.concat(all_tapes)
        results_df = pd.DataFrame(all_results)

        # Add baseline results to summary if generated
        if baseline_results_df is not None:
            results_df = pd.concat([results_df, baseline_results_df])

        # Plot aggregated results for each mix and noise level
        output_dir = os.path.join(experiment_base, 'plots')
        for mix in ['mix_1', 'mix_2']:
            for noise in ['0.00', '0.10', '0.20']:
                plot_aggregated_results(mix, noise, all_avg_balances, all_tapes, output_dir)

        # Plot summary results
        plot_summary_results(results_df, output_dir)

        # Save results
        results_df.to_csv(os.path.join(experiment_base, 'validation_results.csv'), index=False)
        print(results_df.groupby(['noise', 'mix', 'trader_type'])[['profit', 'volatility', 'trades', 'price_std', 'spread_avg', 'avg_balance']].mean())
