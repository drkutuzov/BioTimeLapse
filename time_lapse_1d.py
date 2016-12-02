from collections import defaultdict
from sortedcontainers import SortedSet
from scipy.signal import savgol_filter
from sklearn.metrics import auc
import editdistance as ed
import pandas as pd
import numpy as np
import pylab as plt

Params = ['mean', 'auc_plus', 'max_value', 'max_time', 'max_time_to_peak', 'max_time_to_peak_from_root',
                'auc_minus', 'min_value', 'min_time', 'min_time_to_peak', 'min_time_to_peak_from_root']

class TimeSeries(object):
    """ Class for analyzing time series data
    
    Settings parameters:
    --------------------
    t_start : float,
        start time of an analysis region [sec]
        default: first time point
        
    t_end : float,
        end time of an analysis region [sec]
        default: last time point
        
    baseline_start : float,
        start time of a baseline region [sec]
        default: first time point
        
    baseline_end : 
        end time of a baseline region [sec]
        default: last time point
        
    clip_start : float,
        start time of a discarded region [sec]
        default: False, no clip
        
    clip_end : float,
        end time of a discarded region [sec]
        default: False, no clip
        
    smooth_order : int,
        the power of a local polynomial, used for smoothing data
        default: 0.
        
    smooth_width : int (odd integer),
        the width of a smoothing filter window used -- higher the value -- stronger the smoothing more detail on the filter and parameters here --- https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.savgol_filter.html
        default: 5.
        
    normalization : str,
        'percent' (% of baseline), 'std' (y-value is expressed in the units of the std of a baseline)
        default: 'percent'
        
    clip_interpolate : bool,
        'False' (no interpolation for the clipped region) or any integer (0, 1, 2 ...) which uses polynomial to approximate the data in the clipped region
        default: False, no interpolation
        
    intensity_threshold : float,
        if value of the absolute maximum of a signal (between certain roots) is bigger than intensity_threshold, the parameters are calculated, [% of a baseline]
        default: 5.0% 
        
    duration_threshold : float,
        if the duration of an inter-root interval is longer than duration_threshold, the parameters are calculated [sec]
        default: 10 sec
        
    first_root_init : bool,
        If True, t_start is used as a first root
        default: True
        
    last_root_init : bool,
        If True, t_end is used as a last root
        default: True
    """
    
    def __init__(self, data, settings={}, roots=None, labels=None):
        self.roi_type = data.iloc[0]
        self.name = data.name
        self._y = data.iloc[1:].copy()
        self._t = self._y.index
        self._init_settings(settings)
        self._normalize(self.settings['normalization'])
        self._y = pd.Series(savgol_filter(self._y, self.settings['smooth_width'], self.settings['smooth_order']), 
                            index=self._t)
        self._clip()
        self.roots = self._find_roots() if roots is None else SortedSet(roots)
        if self.settings['first_root_init']:
            self.roots.add(self.settings['t_start'])
        if self.settings['last_root_init']:
            self.roots.add(self.settings['t_end'])
        self._calculate_results()
        self.labels = self._get_interval_labels() if labels is None else labels
    
    def _init_settings(self, settings):
        
        defaults = {
                    't_start' : self._t[0],
                    't_end' : self._t[-1],
                    'baseline_start' : self._t[0],
                    'baseline_end' : self._t[-1],
                    'clip_start' : False,
                    'clip_end' : False,
                    'normalization' : 'percent',
                    'smooth_width' : 5,
                    'smooth_order' : 0,
                    'clip_interpolate' : False,
                    'intensity_threshold' : 5.,
                    'duration_threshold' : 10.,
                    'first_root_init' : True,
                    'last_root_init' : True,
                    }
        
        result = dict(defaults)
        for k, v in settings.iteritems():
            if k not in result:
                candidates = sorted([(word, ed.eval(word, k)) for word in defaults.keys()], key=lambda x: x[1])
                raise Exception('Unknown settings option {}, maybe you meant: {} \n\nAll available parameters are:\n'.format(k, candidates[0][0]) + ', '.join(defaults.keys()))
            result[k] = v                      
        self.settings = result
    
    @property
    def baseline_mean(self):
        return self._y.loc[slice(self.settings['baseline_start'], self.settings['baseline_end'])].mean()
    
    @property
    def baseline_std(self):
        return self._y.loc[slice(self.settings['baseline_start'], self.settings['baseline_end'])].std()
    
    def __getitem__(self, t):
        return self._y.iloc[self._t.get_loc(t, method='nearest')]
    
    def __repr__(self):
        return self._y.__repr__()
    
    def _sel(self, t_start, t_end):
        return self._y.loc[slice(t_start, t_end)]
        
    def _clip(self):
        if self.settings['clip_start'] & self.settings['clip_end']:
            self._y.loc[slice(self.settings['clip_start'], self.settings['clip_end'])] = np.NaN
            if self.settings['clip_interpolate']:
                self._y.interpolate(method='polynomial', order=self.settings['clip_interpolate'], inplace=True)
            
    def _normalize(self, method='percent'):
        if method not in ['percent', 'std']:
            raise Exception('normalization parameter can be either "percent" or "std"')
        self._y = 100 * (self._y / self.baseline_mean - 1) if method == 'percent' \
            else (self._y - self.baseline_mean) / self.baseline_std
    
    def _area_under_curve(self, t1, t2):
        y = self._sel(t1, t2) - self.baseline_mean
        t = y.index - y.index[0]
        if y.isnull().values.any():
            return 'clipped', 'clipped'
        elif len(y) <= 1:
            return 'few data points', 'few data points'
        return auc(t, y.where(y>=0).fillna(0)), auc(t, abs(y.where(y<0).fillna(0)))
        
    def del_root(self, index):
        """ Removes the root and updates results
        
        Parameters:
        -----------
        index : int
            Index of the root to remove (starts from zero)
            
        Example:
        -----------
        y.del_root(2) -- adds a 3rd root in the list (index = 2)
        y -- instance of the TimeSeries class
        """
        del self.roots[index]
        self._calculate_results()
        self.labels = self._get_interval_labels()
    
    def add_root(self, root):
        """ Adds new root and updates results
        
        Parameters:
        -----------
        root : int or float
            Value of a new root (in seconds)
            
        Example:
        -----------
        y.add_root(66.6) -- adds a root to the 66.6th second
        y -- instance of the TimeSeries class
        """
        def check_root(root):
            if (root < self._t[0]) or (root > self._t[-1]):
                raise Exception('A new root value should be between {} and {} sec'.format(self._t[0], self._t[-1]))
            if (root > self.settings['clip_start']) and (root < self.settings['clip_end']):
                raise Exception('A new root value cannot be within the clipped data interval')
        
        check_root(root)
        self.roots.add(root)
        self._calculate_results()
        self.labels = self._get_interval_labels()
        
    def _get_interval_labels(self):
        """ Classifies each inter-root interval to 'dilation' or 'constriction'
        """
        return ['dilation' if abs(y_max-self.baseline_mean) > abs(y_min-self.baseline_mean) else 'constriction' \
                for y_max, y_min in zip(self._results['max_value'], self._results['min_value'])]
        
    def _find_roots(self):
        """ Calculates roots (time points where intensity crosses the baseline)
        """
        d = np.sign(self._sel(self.settings['t_start'], self.settings['t_end']) - self.baseline_mean).diff()
        return SortedSet(np.mean(d.index[i-1:i+1]) for i in np.arange(0, len(d)) 
                           if np.isfinite(d.iloc[i]) & (d.iloc[i] != 0))    
        
    def _calculate_results(self):
        """  Calculates all the parameters of interest from each inter-root interval
        """
        self._results = defaultdict(list)
        funcs = dict(max=(np.argmax, np.max), min=(np.argmin, np.min))
        for t1, t2 in zip(self.roots, self.roots[1:]):
            y = self._sel(t1, t2)
            if (t2 - t1 >= self.settings['duration_threshold']) & (max(abs(y)) >= self.settings['intensity_threshold']): 
                self._results['mean'].append(np.mean(y))
                area_plus, area_minus = self._area_under_curve(t1, t2)
                self._results['auc_plus'].append(area_plus)
                self._results['auc_minus'].append(area_minus)
                for extrema_type, f in funcs.iteritems():
                    self._results[extrema_type+'_time'].append(f[0](y))
                    self._results[extrema_type+'_value'].append(f[1](y))
                    self._results[extrema_type+'_time_to_peak'].append(self._results[extrema_type+'_time'][-1] - self.settings['t_start'])
                    self._results[extrema_type+'_time_to_peak_from_root'].append(self._results[extrema_type+'_time'][-1] - t1)   
            else: 
                for k in Params:
                    self._results[k].append(np.nan)
            
    def get_results(self):
        """ Outputs results as an excel-like table (pandas.DataFrame)
        
        Example:
        --------
        y.get_results(); y -- instance of the TimeSeries class 
        """
        
        times = ['{0:.1f} to {1:.1f} sec'.format(t1, t2) for t1, t2 in zip(self.roots, self.roots[1:])]
        indexes = [(i, j, k) for i, j in zip(self.labels, times) for k in Params]
        return pd.DataFrame(data=np.vstack([self._results[i] for i in Params]).T.ravel(), 
                            index=pd.MultiIndex.from_tuples(indexes, names=['label', 'time', 'parameter']), 
                            columns=(self.name + ' (' + self.roi_type + ')',))
    
    def plot(self, figsize=(20, 5)):
        """ Plots data and calculated parameters
        
        Example:
        --------
        y.plot(); y -- instance of the TimeSeries class

        y.plot(figsize=(25, 8)) -- for the bigger figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel('Time, sec', fontsize=16)
        ax.set_ylabel('Intensity, % of a baseline', fontsize=16)
        ax.set_title(self.name + ' (' + self.roi_type + ')', fontsize=20)
        #---------------------Plotting data----------------------------------------------
        self._y.plot(ax=ax, marker='.', linestyle='-', color='royalblue')
        #---------------------Plotting clipped region------------------------------------
        if self.settings['clip_start'] & self.settings['clip_end']:
            self._sel(self.settings['clip_start'], self.settings['clip_end']).plot(ax=ax, 
                            marker='.', linestyle='-', color='r')
            ax.fill_betweenx([self._y.max(), self._y.min()], self.settings['clip_start'], 
                         self.settings['clip_end'], alpha=.3, color='red')
        #---------------------Plotting Baseline and Stimulation region--------------------
        ax.axhline(y=self.baseline_mean, linewidth=1, linestyle='--', color='k')
        ax.fill_betweenx([self._y.max(), self._y.min()], self.settings['baseline_start'], 
                         self.settings['baseline_end'], alpha=.1, color='royalblue')
        ax.fill_betweenx([self._y.max(), self._y.min()], self.settings['t_start'], 
                         self.settings['t_end'], alpha=.1, color='red')
        #---------------------Plotting calculated peaks-----------------------------------
        ax.plot(self._results['max_time'], self._results['max_value'], 'ro', markersize=10, alpha=.5)
        ax.plot(self._results['min_time'], self._results['min_value'], 'bo', markersize=10, alpha=.5)
        #---------------------Plot roots positions----------------------------------------
        for t in self.roots:
            ax.axvline(x=t, color='k')
            
            
def analyze_table(table, settings, roots, labels, filename):
    """ Analyzes the data in the DataFrame and creates two excel files, figures, containing results
    
    Parameters:
    -----------
    table : DataFrame
        table, which contains data. 
        If the name of the excel sheet is data.xlsx then one can write:
        >>> import pandas as pd
        >>> pd.read_excel(C:/.../data.xlxs, parameters)
        See http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_excel.html for more detail
        
    settings: dict
            settings dictionary
    roots: SortedSet
            Time coordinates of the roots (where y-value intersects a baseline)
    labels : list
            list containing a label for each inter-root interval 
    filename : str
            a full file name for the output (excel sheets and figures)
            example: C:/.../final_folder/file (without .xls extension)

    """
    result_percent, result_std = [], []
    settings['normalization'] = 'percent'
    
    for i in table:
        y = TimeSeries(table[i], settings=settings, roots=roots, labels=labels)
        y.plot()
        plt.savefig(filename + '_' + y.name + '.pdf')
        result_percent.append(y.get_results())
    
    settings['normalization'] = 'std'
    result_std = [TimeSeries(table[i], settings=settings, roots=roots, labels=labels).get_results() for i in table]
        
    writer1 = pd.ExcelWriter(filename + '_percent.xlsx', engine='xlsxwriter')
    pd.concat(result_percent, axis=1).to_excel(writer1, sheet_name='percent')
    writer1.save() 
    writer2 = pd.ExcelWriter(filename + '_STD.xlsx', engine='xlsxwriter')
    pd.concat(result_std, axis=1).to_excel(writer2, sheet_name='STD')
    writer2.save() 