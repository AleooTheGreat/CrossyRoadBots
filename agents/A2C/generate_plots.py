import matplotlib.pyplot as plt
import pandas as pd
import re
import os

# --- Configuration ---
LOG_FILENAME = './medium-hard/a2c_medium-hard_training_log.txt'  # REPLACE this with your log filename
SMOOTHING_WINDOW = 560                       # Window size for moving averages
figure_size = (10, 6)                       # Resolution of saved images

def parse_log_file(filename):
    """Parses the training log file into a pandas DataFrame."""
    data = []
    # Regex to extract metrics
    pattern = re.compile(
        r"Episode (\d+)/\d+, Score: ([-\d\.]+), Reward: ([-\d\.]+), "
        r"Avg: ([-\d\.]+), Actor loss: ([-\d\.]+), Critic loss: ([-\d\.]+), "
        r"Total loss: ([-\d\.]+)"
    )

    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return None

    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                data.append({
                    'Episode': int(match.group(1)),
                    'Score': float(match.group(2)),
                    'Reward': float(match.group(3)),
                    'Average Score': float(match.group(4)),
                    'Actor Loss': float(match.group(5)),
                    'Critic Loss': float(match.group(6)),
                    'Total Loss': float(match.group(7))
                })
    
    return pd.DataFrame(data)

def save_single_plot(x, y1, y1_label, filename, y2=None, y2_label=None, 
                     title='', xlabel='', ylabel='', color1='blue', color2='orange'):
    """Helper function to create and save a single plot."""
    plt.figure(figsize=figure_size)
    
    # Plot primary data (usually noisy/raw)
    plt.plot(x, y1, label=y1_label, alpha=0.4, color=color1)
    
    # Plot secondary data (usually smoothed)
    if y2 is not None:
        plt.plot(x, y2, label=y2_label, color=color2, linewidth=2)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename)
    plt.close() # Close to free up memory
    print(f"Saved {filename}")

def generate_plots(df):
    """Generates and saves individual plots from the DataFrame."""
    if df.empty:
        print("No data found in the log file.")
        return

    # 1. Score Plot
    save_single_plot(
        df['Episode'], df['Score'], 'Score', 'plot_score.png',
        y2=df['Average Score'].rolling(window=SMOOTHING_WINDOW).mean(), y2_label='Avg Score',
        title='Score per Episode', xlabel='Episode', ylabel='Score',
        color1='blue', color2='orange'
    )

    # 2. Reward Plot
    save_single_plot(
        df['Episode'], df['Reward'], 'Reward', 'plot_reward.png',
        y2=df['Reward'].rolling(window=SMOOTHING_WINDOW).mean(), 
        y2_label=f'Reward (MA {SMOOTHING_WINDOW})',
        title='Reward per Episode', xlabel='Episode', ylabel='Reward',
        color1='green', color2='darkgreen'
    )

    # 3. Actor Loss Plot
    save_single_plot(
        df['Episode'], df['Actor Loss'], 'Actor Loss', 'plot_actor_loss.png',
        y2=df['Actor Loss'].rolling(window=SMOOTHING_WINDOW).mean(), 
        y2_label=f'MA {SMOOTHING_WINDOW}',
        title='Actor Loss', xlabel='Episode', ylabel='Loss',
        color1='red', color2='darkred'
    )

    # 4. Critic Loss Plot
    save_single_plot(
        df['Episode'], df['Critic Loss'], 'Critic Loss', 'plot_critic_loss.png',
        y2=df['Critic Loss'].rolling(window=SMOOTHING_WINDOW).mean(), 
        y2_label=f'MA {SMOOTHING_WINDOW}',
        title='Critic Loss', xlabel='Episode', ylabel='Loss',
        color1='purple', color2='indigo'
    )

    # 5. Total Loss Plot
    save_single_plot(
        df['Episode'], df['Total Loss'], 'Total Loss', 'plot_total_loss.png',
        y2=df['Total Loss'].rolling(window=SMOOTHING_WINDOW).mean(), 
        y2_label=f'MA {SMOOTHING_WINDOW}',
        title='Total Loss', xlabel='Episode', ylabel='Loss',
        color1='brown', color2='black'
    )

    # 6. Score Histogram
    plt.figure(figsize=figure_size)
    plt.hist(df['Score'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('plot_score_hist.png')
    plt.close()
    print("Saved plot_score_hist.png")

if __name__ == "__main__":
    df = parse_log_file(LOG_FILENAME)
    if df is not None:
        generate_plots(df)
        print("\nAll plots generated successfully.")