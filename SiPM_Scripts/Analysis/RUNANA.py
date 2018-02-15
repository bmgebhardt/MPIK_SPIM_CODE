import os
import argparse

# LCT5-1 dates = #['0510161','0510162','0510163','0510164','0610161','1010161']
# LVR dates = ['2811164','0112161','0112162','0212161','0212162']
# LCT5/2dates = ['0510162-0','0510162-5','0510162-10','0510162-15','0510162-20','0510162-25','0510162-30','0510162-35']
dates = ['2811162','2111161','1611161','2111162','2811163']


parser = argparse.ArgumentParser(description='Regr:')
parser.add_argument('-r','--regr',type=int,help='-rx or --r x | 0 no regression (first run), 1 with regression line, 2 #currently fixed value')
args = parser.parse_args()

# 0 based on rmsn calc
# 1 based on regression and conversion factor
# 2 based on fixed value (maybe best for first run)


#calcrmsn still unreliable



if args.regr == 1:
	for date in dates:
		print dates
		print 'From regression line'
		os.system('python HAMWaveformReductionSlice.py -r1 -d'+date)


if args.regr == 2:
	print args.regr
	for date in dates:
		print dates
		print 'Fixed Value'

		os.system('python CHECS_HAMWaveformReductionSlice.py -r2 -d'+date) #currently fixed value
		#os.system('python HAMWaveformReductionSlice2.py -r2 -d'+date) #currently fixed value

if args.regr == 0:
	print args.regr
	for date in dates:
		print dates
		print 'No regression , rms calculated'
		print 'CAREFUL EVEN MORE UNRELIABLE'
		os.system('python HAMWaveformReductionSlice.py -r0 -d'+date)