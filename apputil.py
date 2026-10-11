import plotly.express as px
import pandas as pd

URL = ('https://raw.githubusercontent.com/leontoddjohnson/datasets/'
       'main/data/titanic.csv')

AGE_BINS = [-1, 12, 19, 59, 120]
AGE_LABELS = ['Child', 'Teen', 'Adult', 'Senior']


def load_titanic():
    df = pd.read_csv(URL)

    # strip surrounding whitespace
    columns = df.columns.str.strip()

    # spaces and dashes to underscores
    columns = columns.str.replace(r'[ -]+', '_', regex=True)

    # lowercase every name
    df.columns = columns.str.lower()

    return df


def survival_demographics():
    df = load_titanic()

    # bin ages into categories
    df['age_group'] = pd.cut(df['age'], bins=AGE_BINS, labels=AGE_LABELS)

    # group by class, sex, age
    grouped = df.groupby(['pclass', 'sex', 'age_group'], observed=False)

    # count passengers and survivors
    table = grouped.agg(n_passengers=('survived', 'size'),
                        n_survivors=('survived', 'sum'))
    table = table.reset_index()

    # add survival rate
    table['survival_rate'] = table['n_survivors'] / table['n_passengers']

    # sort for readability
    return table.sort_values(['pclass', 'sex', 'age_group'], ignore_index=True)


def predictor_strength():
    df = load_titanic()

    # keep rows with known age
    compare = df.dropna(subset=['age']).copy()

    # bin ages into categories
    compare['age_group'] = pd.cut(compare['age'], bins=AGE_BINS,
                                  labels=AGE_LABELS)

    # overall survival rate
    base = compare['survived'].mean()

    # total variation in survival
    ss_total = ((compare['survived'] - base) ** 2).sum()

    results = []
    for variable in ['pclass', 'sex', 'age_group']:
        # survival within each group
        groups = (compare.groupby(variable, observed=True)['survived']
                  .agg(['size', 'mean']))

        # share of variation explained
        between = (groups['size'] * (groups['mean'] - base) ** 2).sum()
        eta_sq = between / ss_total

        # accuracy from this variable alone
        best = groups['mean'].clip(lower=1 - groups['mean'])
        accuracy = (groups['size'] * best).sum() / len(compare)

        results.append({'variable': variable,
                        'eta_sq': eta_sq,
                        'accuracy': accuracy,
                        'rate_spread': (groups['mean'].max()
                                        - groups['mean'].min())})

    # strongest predictor first
    return (pd.DataFrame(results)
            .sort_values('eta_sq', ascending=False, ignore_index=True))


def visualize_demographic():
    table = predictor_strength()

    # readable axis labels
    table['variable'] = table['variable'].map({'pclass': 'Class',
                                               'sex': 'Sex',
                                               'age_group': 'Age group'})

    fig = px.bar(table, x='variable', y='eta_sq', color='variable',
                 text=table['eta_sq'].map('{:.1%}'.format),
                 labels={'variable': '',
                         'eta_sq': 'Share of survival variation explained'},
                 title='Sex predicts survival far better than class or age')

    # percent axis, no legend
    fig.update_yaxes(tickformat='.0%')
    fig.update_layout(template='simple_white', showlegend=False)

    return fig


def family_groups():
    df = load_titanic()

    # siblings, parents, and self
    df['family_size'] = df['sibsp'] + df['parch'] + 1

    # group by class and family size
    grouped = df.groupby(['pclass', 'family_size'])

    # count passengers and summarize fares
    table = grouped.agg(n_passengers=('fare', 'size'),
                        avg_fare=('fare', 'mean'),
                        min_fare=('fare', 'min'),
                        max_fare=('fare', 'max'))
    table = table.reset_index()

    # sort for readability
    return table.sort_values(['pclass', 'family_size'], ignore_index=True)


def last_names():
    df = load_titanic()

    # text before the comma
    names = df['name'].str.split(',').str[0]

    # drop stray whitespace
    names = names.str.strip()

    return names.value_counts()


def visualize_families():
    table = family_groups()

    fig = px.line(table, x='family_size', y='avg_fare', color='pclass',
                  markers=True,
                  labels={'family_size': 'Family size',
                          'avg_fare': 'Average fare',
                          'pclass': 'Class'},
                  title='Average fare by family size within each class')

    fig.update_layout(template='simple_white')

    return fig


def determine_age_division():
    df = load_titanic()

    # median age of each class
    class_median = df.groupby('pclass')['age'].transform('median')

    # flag the older half
    df['older_passenger'] = df['age'] > class_median

    return df


def visualize_age_division():
    df = determine_age_division()

    # survival by class and age half
    table = (df.groupby(['pclass', 'older_passenger'])['survived']
             .mean()
             .reset_index())

    # readable legend labels
    table['older_passenger'] = table['older_passenger'].map(
        {True: 'Above class median', False: 'At or below'})

    fig = px.bar(table, x='pclass', y='survived', color='older_passenger',
                 barmode='group',
                 labels={'pclass': 'Class', 'survived': 'Survival rate',
                         'older_passenger': ''},
                 title='Survival rate by age relative to class median age')

    # one tick per class, percent axis
    fig.update_xaxes(tickvals=[1, 2, 3])
    fig.update_yaxes(tickformat='.0%', range=[0, 1])
    fig.update_layout(template='simple_white')

    return fig
