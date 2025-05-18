import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Configuration
output_dir = 'validation_plots'
os.makedirs(output_dir, exist_ok=True)

# Conditions (only baseline)
conditions = [('baseline', 'noise_0.00')]
trial_files = []
for mix, noise in conditions:
    path = f'output/baseline/'
    avg_bal_files = glob.glob(f'{path}baseline_mix_1_noise0.00_trial0000_avg_balance.csv')
    if avg_bal_files:
        trial_files.append((mix, noise, avg_bal_files[0]))
    else:
        print(f"No avg_balance.csv found for {mix}/{noise}")

# Initialize results
results = []
for mix, noise, avg_balance_file in trial_files:
    trial_id = os.path.basename(avg_balance_file).replace('_avg_balance.csv', '')
    base_path = f'output/baseline/'

    # Load files
    avg_balance = pd.read_csv(avg_balance_file)
    blotters = pd.read_csv(f'{base_path}{trial_id}_blotters.csv')

    # Define proprietary traders
    prop_traders = ['P00', 'P01']
    trader_types = {tid: 'ZIC' for tid in prop_traders}

    # Extract trades
    trade_counts = {}
    for tid in prop_traders:
        trades = blotters[blotters['tid'] == tid]
        trade_counts[tid] = len(trades)

    # Load tape
    tape_file = f'{base_path}{trial_id}_tape.csv'
    tape = pd.read_csv(tape_file) if os.path.exists(tape_file) else pd.DataFrame({'type': [], 'Time': [], 'price': []})
    price_std = tape['price'].std() if not tape.empty else 0

    # Metrics
    profits = {tid: (avg_balance[tid].iloc[-1] - 10000) for tid in prop_traders}
    volatilities = {tid: avg_balance[tid].std() for tid in prop_traders}
    price_std = tape['price'].std() if not tape.empty else 0

    # Store results
    for tid in prop_traders:
        results.append({
            'trial': trial_id,
            'noise': noise,
            'mix': mix,
            'trader_id': tid,
            'trader_type': trader_types.get(tid, 'Unknown'),
            'profit': profits[tid],
            'volatility': volatilities[tid],
            'trades': trade_counts.get(tid, 0),
            'price_std': price_std
        })

    # Plot 1: Balance over time
    plt.figure(figsize=(10, 6))
    for tid in prop_traders:
        plt.plot(avg_balance['Time'], avg_balance[tid], label=f'{trader_types[tid]} ({tid})')
    plt.title(f'Balance Over Time - {trial_id}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Balance')
    plt.legend()
    plt.savefig(f'{output_dir}/{trial_id}_balance.png')
    plt.close()

    # Plot 2: Trade prices
    plt.figure(figsize=(10, 6))
    plt.scatter(tape['Time'], tape['price'], s=10)
    plt.title(f'Trade Prices - {trial_id}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Price')
    plt.savefig(f'{output_dir}/{trial_id}_prices.png')
    plt.close()

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('validation_results.csv', index=False)

# Summary plots
plt.figure(figsize=(12, 8))
sns.barplot(x='noise', y='trades', hue='trader_type', data=results_df)
plt.title('Trade Frequency by Noise and Trader Type')
plt.savefig(f'{output_dir}/trade_frequency.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.barplot(x='noise', y='profit', hue='trader_type', data=results_df)
plt.title('Profit by Noise and Trader Type')
plt.savefig(f'{output_dir}/profit.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.barplot(x='noise', y='price_std', hue='mix', data=results_df)
plt.title('Price Variance by Noise and Mix')
plt.savefig(f'{output_dir}/price_variance.png')
plt.close()

# Print summary
print(results_df.groupby(['noise', 'mix', 'trader_type'])[['profit', 'volatility', 'trades', 'price_std']].mean())