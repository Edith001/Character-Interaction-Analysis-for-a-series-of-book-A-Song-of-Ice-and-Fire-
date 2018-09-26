# -*- coding: utf-8 -*-


import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt






dataset_path = '../data'


def get_top_records(series_list, top_n=10, show_figure=False):
    """
        pick up the top k important charcter in books
    """
    for i, series in enumerate(series_list):
        print('The {} book has the important {} characters：'.format(i + 1, top_n))
        # pick up the top k points
        top_characters = series.sort_values(ascending=False)[:top_n]
        print(top_characters)

        if show_figure:
            plt.figure(figsize=(10, 8))
            top_characters.plot(kind='bar', title='第{}本书'.format(i + 1))
            plt.tight_layout()
            plt.show()
        print()


def main():
    """
       main function is built here
    """

    # import the data
    print('\n===================== task1. import data =====================')
    book1_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book1-edges.csv'))
    book2_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book2-edges.csv'))
    book3_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book3-edges.csv'))
    book4_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book4-edges.csv'))
    book5_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book5-edges.csv'))

    print(book1_df.head())

    # create the network
    print('\n===================== task2. create the network =====================')
    # create newrok from dataframe
    G_book1 = nx.from_pandas_dataframe(book1_df, 'Source', 'Target', edge_attr=['weight', 'book'])
    G_book2 = nx.from_pandas_dataframe(book2_df, 'Source', 'Target', edge_attr=['weight', 'book'])
    G_book3 = nx.from_pandas_dataframe(book3_df, 'Source', 'Target', edge_attr=['weight', 'book'])
    G_book4 = nx.from_pandas_dataframe(book4_df, 'Source', 'Target', edge_attr=['weight', 'book'])
    G_book5 = nx.from_pandas_dataframe(book5_df, 'Source', 'Target', edge_attr=['weight', 'book'])

    G_books = [G_book1, G_book2, G_book3, G_book4, G_book5]

    # check edges
    print('edges for the first book：')
    print(G_book1.edges(data=True))

    # visualization
    plt.figure(figsize=(10, 9))
    nx.draw_networkx(G_book1)
    plt.show()

    # task3. network analysis
    print('\n===================== task3. network analysis =====================')

    print('\n===================== task3.1 check most important node =====================')
    print('Degree Centrality')
    # caculate degree centrality
    # construct as Series
    deg_cent_list = [nx.degree_centrality(G_book) for G_book in G_books]
    deg_cent_series_list = [pd.Series(deg_cent) for deg_cent in deg_cent_list]
    get_top_records(deg_cent_series_list, show_figure=True)

    print('Closeness Centrality')
    # caculate closeness centrality
    # construct as Series
    clo_cent_list = [nx.closeness_centrality(G_book) for G_book in G_books]
    clo_cent_series_list = [pd.Series(clo_cent) for clo_cent in clo_cent_list]
    get_top_records(clo_cent_series_list, show_figure=True)

    print('Betweenness Centrality')
    # caculate betweenness centrality
    # construct as Series
    btw_cent_list = [nx.betweenness_centrality(G_book) for G_book in G_books]
    btw_cent_series_list = [pd.Series(btw_cent) for btw_cent in btw_cent_list]
    get_top_records(btw_cent_series_list, show_figure=True)

    print('Page Rank')
    # caculate page rank
    # construct the result as Series
    page_rank_list = [nx.pagerank(G_book) for G_book in G_books]
    page_rank_series_list = [pd.Series(page_rank) for page_rank in page_rank_list]
    get_top_records(page_rank_series_list, show_figure=True)

    print('\n===================== task 3.2 relations  =====================')
    cor_df = pd.DataFrame(columns=['Degree Centrality', 'Closeness Centrality', 'Betweenness Centrality', 'Page Rank'])
    cor_df['Degree Centrality'] = pd.Series(nx.degree_centrality(G_book1))
    cor_df['Closeness Centrality'] = pd.Series(nx.closeness_centrality(G_book1))
    cor_df['Betweenness Centrality'] = pd.Series(nx.betweenness_centrality(G_book1))
    cor_df['Page Rank'] = pd.Series(nx.pagerank(G_book1))
    print(cor_df.corr())
    print('\n===================== task 3.3 trend of important chacters =====================')
    trend_df = pd.DataFrame(columns=['Book1', 'Book2', 'Book3', 'Book4', 'Book5'])
    trend_df['Book1'] = pd.Series(nx.degree_centrality(G_book1))
    trend_df['Book2'] = pd.Series(nx.degree_centrality(G_book2))
    trend_df['Book3'] = pd.Series(nx.degree_centrality(G_book3))
    trend_df['Book4'] = pd.Series(nx.degree_centrality(G_book4))
    trend_df['Book5'] = pd.Series(nx.degree_centrality(G_book5))
    trend_df.fillna(0, inplace=True)

    # top10 characters trend in book 1
    top_10_from_book1 = trend_df.sort_values('Book1', ascending=False)[:10]
    top_10_from_book1.T.plot(figsize=(10, 8))
    plt.tight_layout()
    plt.savefig('./role_trend.png')
    plt.show()

    print('\n===================== task3.4 network visualization =====================')
    plt.figure(figsize=(15, 10))

    # degree decides the node color
    node_color = [G_book1.degree(v) for v in G_book1]

    # size of the data is decided by degree centrality
    node_size = [10000 * nx.degree_centrality(G_book1)[v] for v in G_book1]

    # edit the edge_width
    edge_width = [0.2 * G_book1[u][v]['weight'] for u, v in G_book1.edges()]

    # use spring for lay out
    pos = nx.spring_layout(G_book1)

    nx.draw_networkx(G_book1, pos, node_size=node_size,
                     node_color=node_color, alpha=0.7,
                     with_labels=False, width=edge_width)

    # pick up the top 10 important characters in book 1
    top10_in_book1 = top_10_from_book1.index.values.tolist()
    # create label
    labels = {role: role for role in top10_in_book1}

    # add label
    nx.draw_networkx_labels(G_book1, pos, labels=labels, font_size=10)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./book1_network.png')
    plt.show()


if __name__ == '__main__':
    main()
