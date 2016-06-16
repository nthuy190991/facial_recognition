# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:09:31 2016

@author: GGQN0871
"""

from __future__ import print_function
import xlrd

def read_xls(fname, display):

    # Open the workbook
    xl_workbook = xlrd.open_workbook(fname)

    # Grab the first sheet by index
    #  (sheets are zero-indexed)
    xl_sheet = xl_workbook.sheet_by_index(0)
    if display:
        print ('Sheet name: %s' % xl_sheet.name)

    # Pull the first row by index
    #  (rows/columns are also zero-indexed)
    if display:
        row0 = xl_sheet.row(0)  # 1st row
        row  = xl_sheet.row(1)  # 2nd row

        # Print 1st row values and types
        from xlrd.sheet import ctype_text

        print('(Column #) type:value')
        for idx, cell_obj in enumerate(row):
            cell_type_str = ctype_text.get(cell_obj.ctype, 'unknown type')
            print('(%s) %s \t: %s' % (idx, row0[idx].value, cell_type_str))

    # Print all values, iterating through rows and columns
    tb=[['' for i in range(xl_sheet.ncols)] for j in range(xl_sheet.nrows)]

    for row_idx in range(0, xl_sheet.nrows):    # Iterate through rows
        if display:
            print ('-'*40)
            print ('Row: %s' % row_idx)   # Print row number
        for col_idx in range(0, xl_sheet.ncols):  # Iterate through columns
            cell_obj = xl_sheet.cell(row_idx, col_idx)  # Get cell object by row, col
            tb[row_idx][col_idx] = xl_sheet.cell(row_idx, col_idx).value

            if display:
                print ('Column: [%s] cell_obj: [%s]' % (col_idx, cell_obj.value))

    return tb

#if __name__ == "__main__":
#    tb_formation=read_xls('formation',0)
#    #xlrd.xldate_as_tuple(tb_formation[3][5],0)
#    date = xlrd.xldate_as_tuple(tb_formation[3][5],0)
#
#    print "{}/{}/{} ".format(date[2], date[1], date[0])
