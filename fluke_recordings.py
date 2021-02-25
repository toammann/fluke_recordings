import csv
import numpy as np                                     #for csv parser
from datetime import datetime, timezone, timedelta     #for csv parser, num_integrate_avg
from scipy import integrate                            #for trapz() integration


class fluke_recordings:
    """Example Google style docstrings.
    This module provides some tools to deal with fluke trendplots or recordings

    Todo:
        * Implement download from Fluke 289/287
        * Take proper care of units
    """

    data = {}   
    """dict: Data dictonary with keys:  'meas_dev'
                                        'idx'
                                        'samples
                                        'min'
                                        'max'
                                        'avg'
                                        't_start'
                                        't_stop'
                                        'dur'
                                        'desc'
                                        'summary_value'
                                        'summary_sample'
                                        'unit'
    """
    
    # def __init__(self):
    #     self.meas_device =

    def parse(self, filename_csv):
        """
        Parses a *.csv file formatted as a FlukeView Forms export. The parsed data is 
        provided as a member variable 
        Args:
            filename_csv: Filename of the csv file
        """
        fluke_date_fmt_str     = '%d.%m.%Y %H:%M:%S,%f'  #format string for timestamps
        fluke_date_fmt_str_dur = '%H:%M:%S,%f'           #format string for duration

        with open(filename_csv, 'r', encoding='windows-1252') as csv_file:
            
            csv_reader = csv.reader(csv_file, delimiter=";")
            meas_device = next(csv_reader)
            
            idx = np.array([])
            samples =  np.array([])
            t_start = np.array([])
            dur = np.array([])
            t_max = np.array([])
            max_val = np.array([])
            avg = np.array([])
            t_min = np.array([])
            min_val = np.array([])
            desc = []
            t_stop = np.array([])

            #Parse header
            for row in csv_reader:
                if len(row) > 0: #skip empty lines
                    
                    if row[0] == "Start Time": #Search for samples summary
                        summary_sample = [row, next(csv_reader)]

                    elif row[0] == "Max Time": #Search for value summary
                        summary_value = [row, next(csv_reader)]
                        break

            #Parse Data
            for row in csv_reader:
                if len(row) > 0:
                    #search for data section
                    if row[0] != "Reading":  # ignore away column header
                        #End of data section
                        if row[9] == "Logging Stopped":
                            break

                        #Sample idx
                        idx = np.append(idx, float(row[0]))
                        
                        #Samples (measured at t_start)
                        sample_split = row[1].split()[0]
                        samples = np.append(samples, float(sample_split.replace(',', '.')))

                        t = datetime.strptime(row[2], fluke_date_fmt_str)
                        t_start = np.append(t_start, t) #Sample time stamps

                        #Max measures
                        max_val_split = row[5].split()[0]
                        max_val = np.append(max_val, float(max_val_split.replace(',', '.')))               
                        
                        t = datetime.strptime(row[4], fluke_date_fmt_str)
                        t_max = np.append(t_max, t) #Time stap of max_val samples

                        #Min measures
                        min_val_split = row[8].split()[0]
                        min_val = np.append(min_val, float(min_val_split.replace(',', '.')))

                        t = datetime.strptime(row[7], fluke_date_fmt_str)
                        t_min = np.append(t_min, t) #Time stamp of min_val samples

                        # avg measures
                        avg_split = row[6].split()[0]
                        avg = np.append(avg, float(avg_split.replace(',', '.')))

                        #Time stamp of intervall end
                        t = datetime.strptime(row[10], fluke_date_fmt_str)
                        t_stop = np.append(t_stop, t) 

                        #Duration
                        t = datetime.strptime(row[3], fluke_date_fmt_str_dur)
                        t = t - datetime(1900, 1, 1)
                        dur = np.append(dur, t) #Time stamp of min_val samples
                        
                        #Description
                        desc.append(row[9])

        unit = row[1].split()
        #unit = unit[1].join(unit[2])
        unit = unit[1]
        
        self.data = {
            'meas_dev':meas_device,
            'idx':idx,
            'samples':samples,
            'min':[t_min, min_val],
            'max':[t_max, max_val],
            'avg':avg,
            't_start':t_start,
            't_stop':t_stop,
            'dur':dur,              #in seconds
            'desc':desc,
            'summary_value':summary_value,
            'summary_sample':summary_sample,
            'unit':unit
        }

    def num_integrate_avg(self, t_min, t_max):
        """
        Integrates over the "avg" numerical data set using the composite trapezoidal rule.
        Args:
            t_min: Datetime object, start time of integration
            t_max: Datetime object, end time of integration
        """
        conv_utc_vectorized  = np.vectorize(datetime.replace)   #Vectorized: change timezone of daytime object
        timestamp_vectorized = np.vectorize(datetime.timestamp) #Vectorized: Get number of seconds since epoch
        
        #Collect integration data
        avg = self.data["avg"]

        if type(self.data['t_start'][0]) is datetime:
            #Convert integration limits to seconds since epoch
            t = self.data['t_start'] + self.data['dur']/2
            t = conv_utc_vectorized(t, tzinfo=timezone.utc)
            t = timestamp_vectorized(t)

            #Ensure utc time zone
            t_min = t_min.replace(tzinfo=timezone.utc).timestamp()
            t_max = t_max.replace(tzinfo=timezone.utc).timestamp()

        else:
            total_seconds_vectorized  = np.vectorize(timedelta.total_seconds)
            t = self.data['t_start'] + total_seconds_vectorized(self.data['dur']/2)

        #get selected data
        idx_min = np.abs(t - t_min).argmin()
        idx_max = np.abs(t - t_max).argmin()

        #Slice data to index
        avg = avg[idx_min:idx_max+1]
        t = t[idx_min:idx_max+1]

        #Integrate using the composite trapezoidal rule.
        res = integrate.trapz(avg, t)
        res = res/60/60 #convert to hours
        return res

    def mult_const(self, const):
        """
        Muliplies measurment data with a constant
        Args:
            const: Constant value factor
        """
        if not self.data:
            raise RuntimeError('No data available! Call parse() method before modifying data')

        self.data["avg"]     = self.data["avg"]*const
        self.data["min"][1]  = self.data["min"][1]*const
        self.data["max"][1]  = self.data["max"][1]*const
        self.data["samples"] = self.data["samples"]*const

    def rel_time(self):
        """
        Makes time vectors relative (starting from t = UTC epoch)
        """
        if not self.data:
            raise RuntimeError('No data available! Call parse() method before modifying data')
        
        #Convert timedelta to seconds
        total_seconds_vectorized  = np.vectorize(timedelta.total_seconds)

        #self.data["t_start"] = 
        self.data["t_start"] = total_seconds_vectorized(self.data["t_start"] - self.data["t_start"][0])
        self.data["t_stop"] = total_seconds_vectorized(self.data["t_stop"] - self.data["t_stop"][0])
        self.data["min"][0] = total_seconds_vectorized(self.data["min"][0] - self.data["min"][0][0])
        self.data["max"][0] = total_seconds_vectorized(self.data["max"][0] - self.data["max"][0][0])