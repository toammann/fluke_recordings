import argparse                                 # Command line arguments
import matplotlib.pyplot as plt                 # for plots
import matplotlib.dates as dates                # for matlibplot to datetime conversion
from matplotlib.widgets import SpanSelector     # Select a span in matplotlib by mouse
import numpy as np                              # Numpy
from fluke_recordings import fluke_recordings   # fluke_recordings class
from datetime import datetime, timedelta        # Handle datetime objects

tb_num_rows      = 7
tb_num_cols      = 5
tb_box_height    = 1
tb_box_width     = 1
width            = tb_box_width/tb_num_cols
height           = tb_box_height/(tb_num_rows+1)

def onselect(sel_min, sel_max):
    """
    Callback function of SpanSelector matlibplot object
    Args:
        sel_min: Matplotlib date float (number of days since 0001-01-01 UTC, plus 1)
        sel_max: Matplotlib date float (number of days since 0001-01-01 UTC, plus 1)
        
    """

    if not args.relative_t:
        #Matplotlib represents dates using floating point numbers 
        #specifying the number of days since 0001-01-01 UTC, plus 1
        #use the helper functions from matplotlib.dates to convert back to daytime (timezone_utc)
        sel_min = dates.num2date(sel_min)
        sel_max = dates.num2date(sel_max)
    
    #Numerical integration on data
    res = fr.num_integrate_avg(sel_min, sel_max)

    #Initialize static variable at first call
    if not hasattr(onselect, "idx"):
          onselect.idx = 0

    #idx = len(tb.get_celld())/num_cols
    idx = (onselect.idx % tb_num_rows) + 1


    #Copy old table contents to the next row 
    for row in range(idx,1, -1):
        for column in range(0, tb_num_cols):
            tb.add_cell(row, column, width, height, text=tb[row-1,column].get_text()._text)  

    #add new data to cell 1
    tb.add_cell(1, 0, width, height, text=str(onselect.idx))
    if type(sel_min) is datetime:
        tb.add_cell(1, 1, width, height, text=sel_min.strftime("%H:%M:%S"))
        tb.add_cell(1, 2, width, height, text=sel_max.strftime("%H:%M:%S"))
        tb.add_cell(1, 3, width, height, text="{:.3f}".format((sel_max- sel_min).total_seconds()))
    else:
        tb.add_cell(1, 1, width, height, text="{:.3f}".format(sel_min))
        tb.add_cell(1, 2, width, height, text="{:.3f}".format(sel_max))
        tb.add_cell(1, 3, width, height, text="{:.3f}".format(sel_max- sel_min))
    tb.add_cell(1 ,4, width, height, text="{:.3f}".format(res))
    
    #Increment static variable
    onselect.idx += 1

    # Redraw the figure (this was not necessary unitl ? somewhere in 2022 ?)
    fig.canvas.draw()
    fig.canvas.flush_events()

parser = argparse.ArgumentParser(description='Data visualization tool' \
                                            ' for Fluke 287/289 recordings')

#Argparse actions group
group = parser.add_mutually_exclusive_group(required=True) 
group.add_argument('-p', '--plot',
                           action = 'store_true',
                           help = 'Plot data of recording')

group.add_argument('-i', '--integrate',
                            nargs=2,
                            type = float,
                            action = 'append',
                            metavar=('st[s]', 'sp[s]'),
                            help = 'Integrate over a specified time duration in seconds\
                                    starting from 0 (first sample, average dataset)')

group.add_argument('-gi', '--guiint',
                            action = 'store_true', 
                            help = 'Integrate over a mouse selected time duration (average dataset)')

#Argparse optional
parser.add_argument('-d', '--download',
                            action = 'store_true', 
                            help = 'Download data from Fluke 287')

parser.add_argument('-m', '--multiply', 
                            type = float, 
                            help = 'Paramter to specify a constant multiplier. \
                                    May be used to muliply a ampere measurement with a \
                                    constant line voltage to get watts ')


parser.add_argument('-r', '--relative_t', 
                            action = 'store_true', 
                            help = 'Make time vectors relative (start from t=0)')

#Argparse requried
parser.add_argument('file', help = 'Filename of a FlukeView forms *.csv export file') 
args = parser.parse_args()

#Create fluke_recordings objects
fr = fluke_recordings()

#Parse csv file
fr.parse(args.file)

if args.multiply is not None:
    #Muliply data with a constant
    fr.mult_const(args.multiply)

if args.relative_t:
    #Make time vectors relative (start from t=0)
    fr.rel_time()

if args.plot:
    #Plot all parsed parsed data
    if not args.relative_t:
        t = fr.data['t_start'] + fr.data['dur']/2
    else:
        total_seconds_vectorized  = np.vectorize(timedelta.total_seconds)
        t = fr.data['t_start'] + total_seconds_vectorized(fr.data['dur']/2)

    plt.plot(t, fr.data['avg'], label = 'avg') 
    plt.plot(fr.data['t_start'], fr.data['samples'], label = 'sample')
    plt.plot(fr.data['min'][0], fr.data['min'][1], label = 'min')   
    plt.plot(fr.data['max'][0], fr.data['max'][1], label = 'max')   

    #Configure plot appearance
    plt.gcf().autofmt_xdate()
    plt.grid(True)
    plt.legend()
    plt.xlabel('time')
    
    #Display units
    if args.multiply is not None:
        plt.ylabel(fr.data['unit']+ "V")
    else:
        plt.ylabel(fr.data['unit'])

    plt.show()

elif args.guiint:

    #Create a subplot, displaying the average data set
    fig, (ax, ax_t) = plt.subplots(
        ncols=1,
        nrows=2,
        figsize=(10,8), 
        gridspec_kw={'width_ratios': [1], 'height_ratios': [2, 1]})

    fig.suptitle('Average data set of recording: ' + 
                 fr.data["summary_sample"][1][-1]  + 
                 ", (" + fr.data["meas_dev"][0] + "," +
                 fr.data["meas_dev"][1] + ")",
                 fontsize=14)

    if not args.relative_t:
        t = fr.data['t_start'] + fr.data['dur']/2
        ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
    else:
        total_seconds_vectorized  = np.vectorize(timedelta.total_seconds)
        t = fr.data['t_start'] + total_seconds_vectorized(fr.data['dur']/2)

    ax.plot(t, fr.data['avg'], label = 'avg') 
    ax.tick_params('x', labelrotation=45)
    ax.grid(True)
    #ax.set_xlabel('time')

    #Create spanSlector widget
    span = SpanSelector(
        ax,
        onselect = onselect,
        direction = 'horizontal',
        minspan = 0,
        useblit = True,
        interactive = True,
        button = 1, #left mouse button
        props = {'facecolor':'grey', 'alpha':0.3}
    )

    #Set Axes
    if args.multiply is not None:
        ax.set_ylabel(fr.data['unit']+ "V")
    else:
        ax.set_ylabel(fr.data['unit'])
    ax.legend()

    #Disable Axes for table subplot
    ax_t.get_xaxis().set_visible(False)
    ax_t.get_yaxis().set_visible(False)   
    
    #Create row headings
    if args.multiply is not None:
        int_str = "Integrated [" + fr.data['unit'] +"Vh]"
    else:
        int_str= "Integrated [" + fr.data['unit'] +"h]"

    #Row headers
    c_headers = [ "Index", "Sel. start time",
                  "Sel. end time", "Sel.  duration [s]",
                  int_str]

    #c_colors = plt.cm.BuPu(np.full(len(c_headers), 0.1))
    c_colors = np.full(len(c_headers), 'whitesmoke')

    #Fill table wit empty data
    cell_text = []
    for i in range(tb_num_rows):
         cell_text.append([" ", " ", " "," "," "])


    #Draw table
    tb = ax_t.table(    cellText = cell_text,
                        colColours = c_colors,
                        colLabels  = c_headers,
                        bbox=[0, 0, tb_box_width, tb_box_height], #[left, bottom, width, height]
                        loc='top')

    #Scale table
    #tb.scale(1, 1)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.tight_layout() #Fix spacings between subpltos for cleaner appearance
    plt.show()

elif args.integrate:
    if not args.relative_t:
        #Make time vectors relative if the user did not call with "-r"
        fr.rel_time()

    #Get integration boundaries
    t_start, t_stop = args.integrate[0]

    if t_start < 0:
        print("Start time must be >= 0")
        exit()
    if t_stop > fr.data["t_stop"][-1]:
        print("End time exceeds maximum data time of : ", fr.data["t_stop"][-1],  "s")
        exit()

    #Perform integration
    res = fr.num_integrate_avg(t_start, t_stop)
    print(res)