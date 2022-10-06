import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import operator
from itertools import product
from sklearn.preprocessing import StandardScaler


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
import graphviz

def set_palette():
    # Create an array with the colors you want to use
    colors = ["#5DD5BF", "#5b9ad5", "#000000", "#D56F5D", "#D5BF5D"]
    # Save a palette to a variable:
    uncpalette = sns.color_palette(colors)
    # Set your custom color palette
    sns.set_palette(uncpalette)

def set_style(tick_color="#444",
              grid_color="#ddd",
              tick_width="0.6",
              figure_size=(4, 4)):

    sns.set(style="whitegrid", rc={
        "axes.grid.axis": "y",
        "axes.grid.which": "major",
        "grid.color": grid_color,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.color": tick_color,
        "xtick.color": tick_color,
        "xtick.major.width": tick_width,
        "ytick.major.width": tick_width,
        "xtick.minor.width": tick_width,
        "ytick.minor.width": tick_width,
        "grid.linewidth": tick_width,
        "text.color": "black"
    })
    sns.set_context("paper")
    sns.set_palette(uncpalette)

    SMALL_SIZE = 15
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class GenerateDataframe:
    def __init__(self, input_case, output_case):
        """Creates an object that stores the input and output dataframes that contain simulation results. These objects have a number of 
        methods that are helpful for scenario discovery analysis. Be careful not to mix incompatible input/output cases together (e.g. 
        using U.S. input data to analyze share of renewables in China).

        Args:
            input_case (str): the input data to be used (supported: GLB_GRP_NORM (global, grouped, and normed); GLB_RAW 
            (global, full number of variables, no normalization); USA (full number of variables with GDP + Pop specific to US);
            CHN (full number of variables with GDP + Pop specific to China); EUR (full number of variables with GDP + Pop
            specific to EU))

            output_case (str): the output metric (supported: REF_GLB_RENEW_SHARE (global share of renewables under
            the reference scenario); REF_USA_RENEW_SHARE (US share of renewables under the reference scenario); REF_CHN_RENEW_SHARE
            (China share of renewables under the reference scenario); REF_EUR_RENEW_SHARE (EU share of renewables under the reference
            scenario); 2C_GLB_RENEW_SHARE (global share of renewables under the policy scenario); 2C_USA_RENEW_SHARE (US share of 
            renewables under the policy scenario); 2C_CHN_RENEW_SHARE (China share of renewables under the policy scenario); 
            2C_EUR_RENEW_SHARE (EU share of renewables under the policy scenario); REF_GLB_RENEW (global renewable energy
            production in Twh); REF_GLB_TOT (total global energy production in Twh))

        Raises:
            ValueError: if an invalid input case is passed
            ValueError: if an invalid output metric is passed
        """
        self.input_case = input_case
        self.output_case = output_case

        self.supported_input_scenarios = ['GLB_GRP_NORM', 'GLB_RAW', 'USA', 'CHN', 'EUR']
        self.natural_to_code_conversions_dict_inputs = {'GLB_GRP_NORM': ['samples-norm+groupedav', 'A:J'], 'GLB_RAW': ['samples', 'A:BB'], 'USA': ['samples', 'A:AZ, BC:BF'],
            'CHN': ['samples', 'A:AZ, BG:BJ'], 'EUR': ['samples', 'A:AZ, BK: BN']}
        if self.input_case in self.supported_input_scenarios:
            sheetname = self.natural_to_code_conversions_dict_inputs[self.input_case][0]
            columns = self.natural_to_code_conversions_dict_inputs[self.input_case][1]
            if self.input_case == "GLB_GRP_NORM":
                self.input_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = columns, nrows = 400)
            else:
                self.input_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = columns, nrows = 400, header = 2)
            if '2C' in output_case:
                # indicates a policy scenario in which runs crashed
                crashed_run_numbers = [3, 14, 74, 116, 130, 337]
                self.input_df = self.input_df.drop(index = [i - 1 for i in crashed_run_numbers])
        else:
            raise ValueError('This input scenario is not supported. Supported scenarios are {}'.format(self.supported_input_scenarios))

        self.supported_output_scenarios = ['REF_GLB_RENEW_SHARE', 'REF_USA_RENEW_SHARE', 'REF_CHN_RENEW_SHARE', 'REF_EUR_RENEW_SHARE',
            '2C_GLB_RENEW_SHARE', '2C_USA_RENEW_SHARE', '2C_CHN_RENEW_SHARE', '2C_EUR_RENEW_SHARE', 'REF_GLB_RENEW', 'REF_GLB_TOT']
        self.natural_to_code_conversions_dict_outputs = {'REF_GLB_RENEW_SHARE': 'ref_GLB_renew_share', 'REF_USA_RENEW_SHARE': 'ref_USA_renew_share',
            'REF_CHN_RENEW_SHARE': 'ref_CHN_renew_share', 'REF_EUR_RENEW_SHARE': 'ref_EUR_renew_share', '2C_GLB_RENEW_SHARE': '2C_GLB_renew_share', 
            '2C_USA_RENEW_SHARE': '2C_USA_renew_share', '2C_CHN_RENEW_SHARE': '2C_CHN_renew_share', '2C_EUR_RENEW_SHARE': '2C_EUR_renew_share',
            'REF_GLB_RENEW': 'ref_GLB_renew', 'REF_GLB_TOT': 'ref_GLB_total_elec'}
        if self.output_case in self.supported_output_scenarios:
            sheetname = self.natural_to_code_conversions_dict_outputs[output_case]
            self.output_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = 'D:X', nrows = 400)
            if '2C' in output_case:
                # indicates a policy scenario in which runs crashed
                crashed_run_numbers = [3, 14, 74, 116, 130, 337]
                self.output_df = self.output_df.drop(index = [i - 1 for i in crashed_run_numbers])
        else:
            raise ValueError('This output scenario is not supported. Supported scenarios are {}'.format(self.supported_output_scenarios))

    def get_X(self):
        """Get the exogenous dataset (does not include run numbers).

        Returns:
            DataFrame: Input variables and their values
        """
        return self.input_df[self.input_df.columns[1:]]

    def get_y(self):
        """Get the endogenous dataset (does not include run numbers).

        Returns:
            DataFrame: Output timeseries
        """
        return self.output_df[self.output_df.columns[1:]]

    def get_y_by_year(self, year):
        """Get the series for an individual year.

        Args:
            year (int): A year included in the dataset (options:
            2007, and 2010-2100 in 5-year increments)

        Returns:
            Series: A pandas Series object with the data from the given year
        """
        return self.output_df[str(year)]
    
    def parallel_plot(self, year, percentile = 70):
        import plotly.graph_objects as go
        import plotly.express as px
    
        generator_for_plot = GenerateDataframe("GLB_GRP_NORM", self.output_case)
        X = generator_for_plot.get_X()
        y = generator_for_plot.get_y_by_year(year)
        perc = np.percentile(y, percentile)

        dataframe_for_plot = X.copy()
        dataframe_for_plot['metric'] = y
        dataframe_for_plot_sorted = dataframe_for_plot.sort_values(by = ['metric'])
        y_discrete = np.where(dataframe_for_plot_sorted['metric'] > perc, 1, 0)

        dimensions_list = []
        for name, data in dataframe_for_plot_sorted.iteritems():
            data_max = max(data)
            data_min = min(data)
            series_dict = dict(range = [data_min, data_max], label = name, values = data)
            dimensions_list.append(series_dict)

        fig = go.Figure(data = go.Parcoords(line = dict(color = y_discrete,
                        colorscale = [[0, 'purple'], [1, 'gold']], showscale = True), dimensions = dimensions_list))
        

        fig.update_layout(
            plot_bgcolor = 'white',
            paper_bgcolor = 'gray'
        )

        fig.show()

    def CART(self, year, percentile = 70, max_leaf_nodes = 5):
        from sklearn.tree import export_graphviz

        X = self.get_X()
        y = self.get_y_by_year(year)

        perc = np.percentile(y, percentile)
        y_discrete = np.where(y > perc, 1, 0)

        tree = DecisionTreeClassifier(max_leaf_nodes = max_leaf_nodes)
        tree_fit = tree.fit(X, y_discrete)

        print("FEATURE IMPORTANCES\n")
        for name, importance in zip(X.columns, tree_fit.feature_importances_):
            print(name + ': %0.2f' % importance)
        
        dot_data = export_graphviz(tree_fit, 
                  feature_names = X.columns,  
                  class_names = ['Low Renew.', 'High Renew.'],  
                  filled = True, rounded = True,  
                  special_characters = True,
                   out_file = None,
                           )
        graph = graphviz.Source(dot_data)
        graph.format = 'svg'
        graph.render(filename = self.input_case + '; ' + self.output_case + ' ' + str(year))