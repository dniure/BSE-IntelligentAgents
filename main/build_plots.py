import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_trial_results(base_path, trial_id, avg_balance, tape, prop_traders, trader_types):
    plots_dir = os.path.join(base_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Balance over time for prop traders
    plt.figure(figsize=(10, 6))
    for tid in prop_traders:
        plt.plot(avg_balance['time'], avg_balance[tid], label=f'{trader_types[tid]} ({tid})')
    plt.title(f'Balance Over Time - {trial_id}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Balance')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{trial_id}_balance.png'))
    plt.close()

    # Plot 2: Trade prices
    plt.figure(figsize=(10, 6))
    plt.scatter(tape['time'], tape['price'], s=10)
    plt.title(f'Trade Prices - {trial_id}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Price')
    plt.savefig(os.path.join(plots_dir, f'{trial_id}_prices.png'))
    plt.close()

    # Plot 3: Bid-ask spread over time
    plt.figure(figsize=(10, 6))
    plt.plot(avg_balance['time'], avg_balance['spread'], label='Spread', color='red')
    plt.title(f'Bid-Ask Spread Over Time - {trial_id}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spread')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{trial_id}_spread.png'))
    plt.close()

    # Plot 4: ZIC Buyer/Seller balances over time
    plt.figure(figsize=(10, 6))
    plt.plot(avg_balance['time'], avg_balance['Buyer_ZIC_Balance'], label='Buyer ZIC Balance', color='blue')
    plt.plot(avg_balance['time'], avg_balance['Seller_ZIC_Balance'], label='Seller ZIC Balance', color='green')
    plt.title(f'Buyer/Seller ZIC Balance Over Time - {trial_id}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Balance')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{trial_id}_buyer_seller_balance.png'))
    plt.close()

    # Plot 5: LOB prices over time
    plt.figure(figsize=(10, 6))
    plt.plot(avg_balance['time'], avg_balance['bidPrice'], label='Bid Price', color='blue')
    plt.plot(avg_balance['time'], avg_balance['askPrice'], label='Ask Price', color='red')
    plt.plot(avg_balance['time'], avg_balance['midPrice'], label='Mid Price', color='green')
    plt.title(f'LOB Prices Over Time - {trial_id}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{trial_id}_lob_prices.png'))
    plt.close()

def plot_summary_results(results_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Summary Plot 1: Trade frequency by noise and trader type (prop traders)
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='noise', y='trades', hue='trader_type', data=results_df[results_df['trader_id'].str.startswith('P')])
    plt.title('Trade Frequency by Noise and Trader Type (Prop Traders)')
    plt.savefig(os.path.join(output_dir, 'trade_frequency.png'))
    plt.close()

    # Summary Plot 2: Profit by noise and trader type (prop traders)
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='noise', y='profit', hue='trader_type', data=results_df[results_df['trader_id'].str.startswith('P')])
    plt.title('Profit by Noise and Trader Type (Prop Traders)')
    plt.savefig(os.path.join(output_dir, 'profit.png'))
    plt.close()

    # Summary Plot 3: Price variance by noise and mix
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='noise', y='price_std', hue='mix', data=results_df)
    plt.title('Price Variance by Noise and Mix')
    plt.savefig(os.path.join(output_dir, 'price_variance.png'))
    plt.close()

    # Summary Plot 4: Spread by mix and noise
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='noise', y='spread_avg', hue='mix', data=results_df)
    plt.title('Spread Summary by Mix and Noise')
    plt.savefig(os.path.join(output_dir, 'spread_summary.png'))
    plt.close()

    # Summary Plot 5: Buyer/Seller trade frequency
    plt.figure(figsize=(12, 8))
    buyer_seller_df = results_df[results_df['trader_id'].str.startswith(('B', 'S'))]
    sns.barplot(x='noise', y='trades', hue='trader_id', data=buyer_seller_df)
    plt.title('Buyer/Seller Trade Frequency by Noise')
    plt.savefig(os.path.join(output_dir, 'buyer_seller_trade_frequency.png'))
    plt.close()

    # Summary Plot 6: Buyer/Seller average balance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='noise', y='avg_balance', hue='trader_id', data=buyer_seller_df)
    plt.title('Buyer/Seller Average Balance by Noise')
    plt.savefig(os.path.join(output_dir, 'buyer_seller_avg_balance.png'))
    plt.close()

if __name__ == "__main__":
    trial_id = "baseline_mix_1_noise0.00_trial0000"
    base_path = 'output/baseline/'
    avg_balance = pd.read_csv(f'{base_path}{trial_id}_avg_balance.csv')
    blotters = pd.read_csv(f'{base_path}{trial_id}_blotters.csv')
    tape = pd.read_csv(f'{base_path}{trial_id}_tape.csv')

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
            initial_balance = 10000  # Assuming initial balance
            avg_balances[tid] = avg_balance[tid].iloc[-1] - initial_balance  # Correct profit

    spread_avg = avg_balance['spread'].mean() if not avg_balance.empty else 0

    results = []
    for tid in trade_counts.keys():
        trader_type = 'ZIC' if tid.startswith(('B', 'S')) else 'SHVR'
        results.append({
            'trial': trial_id,
            'noise': '0.00',
            'mix': 'baseline',
            'trader_id': tid,
            'trader_type': trader_type,
            'profit': avg_balances[tid] if tid.startswith('P') else None,
            'volatility': avg_balance[tid].std() if tid in avg_balance.columns else None,
            'trades': trade_counts[tid],
            'price_std': tape['price'].std() if not tape.empty else 0,
            'spread_avg': spread_avg,
            'avg_balance': avg_balances[tid] if tid.startswith(('B', 'S')) else None
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv('validation_results.csv', index=False)

    # Process all trials in the directory
    all_results = []
    for file in os.listdir(base_path):
        if file.endswith('_avg_balance.csv') and 'trial' in file:
            trial_id = file.replace('_avg_balance.csv', '')
            avg_balance = pd.read_csv(os.path.join(base_path, file))
            blotters = pd.read_csv(os.path.join(base_path, trial_id + '_blotters.csv'))
            tape = pd.read_csv(os.path.join(base_path, trial_id + '_tape.csv'))
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

            trial_results = []
            for tid in trade_counts.keys():
                noise = float(trial_id.split('noise')[1].split('_')[0])
                mix = 'baseline' if 'baseline' in trial_id else 'mix_1' if 'mix_1' in trial_id else 'mix_2'
                trader_type = 'ZIC' if tid.startswith(('B', 'S')) else 'SHVR'
                trial_results.append({
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
            all_results.extend(trial_results)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('validation_results.csv', index=False)

    plot_trial_results(base_path, trial_id, avg_balance, tape, prop_traders, trader_types)
    plot_summary_results(results_df, os.path.join(base_path, 'plots'))

    print(results_df.groupby(['noise', 'mix', 'trader_type'])[['profit', 'volatility', 'trades', 'price_std', 'spread_avg', 'avg_balance']].mean())