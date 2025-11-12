import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plot_hist_cloud(file_path=r'Data/COVIDSenti-main/A_ready_lemma_res_A.csv'):

    data = pd.read_csv(file_path)

    name = file_path.split('/')[-1]

    words = []
    data['processed'].dropna().apply(lambda x: words.extend(x.split())) 

    word_counts = Counter(words)

    word_freq_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.bar(word_freq_df['Word'][:20], word_freq_df['Frequency'][:20]) 
    ax1.set_xticklabels(word_freq_df['Word'][:20], rotation=45)
    ax1.set_xlabel('Words')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Top 20 Word Frequencies - {name}')

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.set_title(f'Word Cloud - {name}')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()