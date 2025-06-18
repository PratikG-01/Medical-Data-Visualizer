import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load dataset
df = pd.read_csv('medical_examination.csv')

# 2. Add 'overweight' column using BMI calculation (weight / height^2)
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3. Normalize cholesterol and glucose levels (1 -> 0, else -> 1)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5. Reshape dataframe for categorical plot using melt
    df_cat = pd.melt(
        df,
        id_vars='cardio',
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. Group and count data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size().rename(columns={'size': 'total'})

    # 7. Plot bar chart using seaborn.catplot
    plot = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    )

    # 8. Extract figure object
    fig = plot.fig

    # 9. Save figure to file
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11. Filter out invalid blood pressure values and height/weight outliers
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'].between(df['height'].quantile(0.025), df['height'].quantile(0.975))) &
        (df['weight'].between(df['weight'].quantile(0.025), df['weight'].quantile(0.975)))
    ]

    # 12. Compute correlation matrix
    corr = df_heat.corr()

    # 13. Create mask to hide upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Initialize matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15. Draw heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.5},
        center=0,
        ax=ax
    )

    # 16. Save heatmap to file
    fig.savefig('heatmap.png')
    return fig
