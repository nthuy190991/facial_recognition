# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:54:36 2016

@author: GGQN0871
"""

from xlrd import open_workbook
from xlutils.copy import copy

def edit_xls(fname, row, col, value):
    
    # Open the workbook
    rb = open_workbook(fname)
    wb = copy(rb)

    s = wb.get_sheet(0)
    #s = get_sheet_by_name(wb, 'Sheet1')

    s.write(row, col, value)

    wb.save(fname)

#edit_xlsx('formation2',0,2,12)
