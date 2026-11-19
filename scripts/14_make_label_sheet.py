"""Blinded labelling workbook. usage: 14_make_label_sheet.py REF_DIR
Only location and date go in: no prediction, stratum, DWR or PVID information."""
import sys, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
ref=sys.argv[1]; K=pd.read_csv(f'{ref}/key/DO_NOT_OPEN_key.csv').sort_values('sample_id')
N=len(K); F='Arial'; f=lambda **k:Font(name=F,size=k.pop('size',10),**k); YEL=PatternFill('solid',fgColor='FFF2A8'); HEAD=PatternFill('solid',fgColor='1F3A5F'); GREY=PatternFill('solid',fgColor='EFEFEF'); thin=Side(style='thin',color='CCCCCC'); B=Border(left=thin,right=thin,top=thin,bottom=thin)
wrap=Alignment(wrap_text=True,vertical='top'); LBL=['Fallow','Part-year fallow','Cropped','Not cropland','Cannot tell']; CONF=['High','Medium','Low']
wb=Workbook(); I=wb.active; I.title='Instructions'; L=wb.create_sheet('Labels')
heads=['Sample ID','Water year','Period covered','Open image sheet','Acres','Latitude','Longitude','Google Maps','Label','Longest bare stretch (months)','Confidence','Notes']
for j,h in enumerate(heads,1):
    c=L.cell(1,j,h); c.font=f(bold=True,color='FFFFFF'); c.fill=HEAD; c.alignment=Alignment(wrap_text=True,vertical='center',horizontal='center'); c.border=B
for i,r in enumerate(K.itertuples(),2):
    y=int(r.wy); vals=[r.sample_id,str(y),f'Oct {y-1} to Sep {y}',None,float(r.acres),float(r.lat),float(r.lon),None,None,None,None,None]
    for j,v in enumerate(vals,1):
        c=L.cell(i,j,v); c.font=f(); c.border=B
    c=L.cell(i,4,'Open image'); c.hyperlink=f'chips/{r.sample_id}.png'; c.font=f(color='0563C1',underline='single')
    c=L.cell(i,8,'Open map'); c.hyperlink=f'https://www.google.com/maps/@{r.lat},{r.lon},700m/data=!3m1!1e3'; c.font=f(color='0563C1',underline='single')
    L.cell(i,5).number_format='0.0'; L.cell(i,6).number_format='0.00000'; L.cell(i,7).number_format='0.00000'
    for j in (9,10,11,12): L.cell(i,j).fill=YEL
for col,w in zip('ABCDEFGHIJKL',[10,9,20,14,8,11,12,12,18,13,12,42]): L.column_dimensions[col].width=w
L.row_dimensions[1].height=44; L.freeze_panes='B2'; L.auto_filter.ref=f'A1:L{N+1}'
d1=DataValidation(type='list',formula1='"'+','.join(LBL)+'"',allow_blank=True,showErrorMessage=True,errorTitle='Label',error='Choose one of the listed labels.'); L.add_data_validation(d1); d1.add(f'I2:I{N+1}')
d2=DataValidation(type='whole',operator='between',formula1='0',formula2='12',allow_blank=True,showErrorMessage=True,errorTitle='Months',error='Enter a whole number from 0 to 12.'); L.add_data_validation(d2); d2.add(f'J2:J{N+1}')
d3=DataValidation(type='list',formula1='"'+','.join(CONF)+'"',allow_blank=True,showErrorMessage=True,errorTitle='Confidence',error='Choose High, Medium or Low.'); L.add_data_validation(d3); d3.add(f'K2:K{N+1}')
I.column_dimensions['A'].width=24; I.column_dimensions['B'].width=92
def put(r,a,b=None,bold=False,fill=None):
    c=I.cell(r,1,a); c.font=f(bold=bold or b is not None); c.alignment=wrap
    if fill: c.fill=fill
    if b is not None:
        d=I.cell(r,2,b); d.font=f(); d.alignment=wrap
        if fill: d.fill=fill
r=1; I.cell(r,1,'Palo Verde fallow reference labels').font=f(bold=True,size=15); r+=1
I.cell(r,1,'Label each field from its images alone. Results are used to measure how accurate the fallow maps really are.').font=f(color='52514E'); r+=2
I.cell(r,1,'Progress').font=f(bold=True,size=12); r+=1
I.cell(r,1,'Fields labelled').font=f(); I.cell(r,2,f'=COUNTA(Labels!I2:I{N+1})').font=f(); I.cell(r,2).alignment=Alignment(horizontal='left'); r+=1
I.cell(r,1,'Fields in total').font=f(); I.cell(r,2,f'=COUNTA(Labels!A2:A{N+1})').font=f(); I.cell(r,2).alignment=Alignment(horizontal='left'); r+=1
I.cell(r,1,'Percent complete').font=f(); I.cell(r,2,f'=IFERROR(B{r-2}/B{r-1},0)').font=f(); I.cell(r,2).number_format='0%'; I.cell(r,2).alignment=Alignment(horizontal='left'); r+=2
I.cell(r,1,'Which cells to edit').font=f(bold=True,size=12); r+=1
c=I.cell(r,1,'Yellow cells'); c.fill=YEL; c.font=f(bold=True); d=I.cell(r,2,'On the Labels tab, fill in only the yellow columns: Label, Longest bare stretch, Confidence, and Notes if useful.'); d.font=f(); d.alignment=wrap; r+=2
I.cell(r,1,'How to label a field').font=f(bold=True,size=12); r+=1
for a,b in [('1. Open the image','Click Open image on the Labels tab. Each sheet shows the field outlined in yellow for the 12 months of its water year, plus an aerial photo. Check the date printed above the photo: it can fall outside the water year, so use it for field layout and always judge the crop from the monthly images. Keep the chips folder beside this workbook, or the links will not open.'),('2. Judge each month','Decide whether a crop was growing in each month. Green usually means crop, but bare ground after a harvest does not mean fallow.'),('3. Choose a label','Use the definitions below.'),('4. Count months','Enter the longest run of consecutive months with no crop, from 0 to 12.'),('5. Rate confidence','Say Low if clouds, missing months or weeds make it hard to judge.')]: put(r,a,b); r+=1
r+=1; I.cell(r,1,'Label definitions').font=f(bold=True,size=12); r+=1
for a,b in [('Fallow','No crop was grown during the water year. The field is bare or has only scattered weeds throughout.'),('Part-year fallow','A crop was grown for part of the year, and the field also has at least 6 consecutive months with no crop.'),('Cropped','Crops were present through the year, with no stretch of 6 or more months without a crop. Short gaps between crops still count as Cropped. Permanent crops such as alfalfa count as Cropped.'),('Not cropland','The outlined area is not farmland, such as buildings, a road, a canal or natural vegetation.'),('Cannot tell','The images do not allow a judgement. Use this rather than guessing.')]: put(r,a,b); r+=1
r+=1; I.cell(r,1,'Rules that keep the test fair').font=f(bold=True,size=12); r+=1
for a,b in [('Do not open the key','Do not open the key folder, the fallow maps, or DWR, CDL or PVID layers while labelling. Knowing the predictions would bias the labels.'),('Google Maps','The Open map link shows current imagery, not the water year. Use it only to see the location and setting.'),('Repeat fields','A few fields appear in more than one water year. Label each row from its own images.')]: put(r,a,b); r+=1
r+=1; I.cell(r,1,'Example (not a real sample, do not copy)').font=f(bold=True,size=12); r+=1
for j,h in enumerate(['Sample ID','Label / Longest bare stretch / Confidence / Notes'],1):
    c=I.cell(r,j,h); c.font=f(bold=True); c.fill=GREY
r+=1; I.cell(r,1,'S000 (example)').font=f(); d=I.cell(r,2,'Part-year fallow  /  7  /  High  /  Green Oct to Feb, then bare from Mar to Sep'); d.font=f(); d.alignment=wrap
wb.save(f'{ref}/reference_labels.xlsx'); print('saved',N)
