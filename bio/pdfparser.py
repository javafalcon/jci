# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:22:39 2019

@author: Administrator
"""

import os.path
from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal, LAParams
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed

def parse(pdffile, txtfile):
    '''解析PDF文本，并保存到txt文件中'''
    fp = open(pdffile, 'rb')
    parser = PDFParser(fp) #创建一个PDF分析器
    
    #创建PDF文档，连接分析器与文档对象
    doc = PDFDocument()
    parser.set_document(doc)
    doc.set_parser(parser)
    
    #提供初始化密码，如果没有密码，就创建一个空的字符串
    doc.initialize()
    
    #检测文档是否提供Txt转换，不提供就忽略
    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        #创建PDF资源管理器
        rsrcmgr = PDFResourceManager()
        #创建一个PDF设备对象
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        
        #遍历列表，每次处理一个page内容
        for page in doc.get_pages():
            interpreter.process_page(page)
            #接受该页面的LTPage对象
            layout = device.get_result()
            #layout是一个LTPage对象，里面存放这这个page解析出的各种对象
            #一般包括LTTextBox，LTFigure，LTImage，LTTextBoxHorizontal等等
            #想要获取文本就获得对象的text属性
            for x in layout:
                if(isinstance(x, LTTextBoxHorizontal)):
                    with open(txtfile,'a') as f:
                        results = x.get_text()
                        print(results)
                        f.write(results + "\n")

parse(r'E:\Repoes\jcilwz\RemoteHomology\program\12859_2017_1842_MOESM1_ESM.pdf',
      r'E:\Repoes\jcilwz\RemoteHomology\program\independ_test.txt')       