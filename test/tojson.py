import pandas as pd
def sale_data_month():
    datapath = "D:\\gitRepo\\notebook\\data\\10年原始数据.xlsx"
    df = pd.read_excel(datapath)
    array = []
    for i in range(0, df.shape[0], 2):
        array.append([df["年\月"][i], df["总量"][i]])
    return "[" + ",".join(["[\"{}\",{}]".format(i[0], i[1]) for i in array]) + "]"
def sale_data():
    datapath="D:\\gitRepo\\notebook\\data\\10年原始数据.xlsx"
    df = pd.read_excel(datapath)
    array=[]
    for i in range(0, df.shape[0], 2):
        for j in range(1,13):
            row = str(j)+"月"
            array.append([df["年\月"][i]+" "+row,df[row][i]])

    return "["+",".join(["[\"{}\",{}]".format(i[0],i[1]) for i in array])+"]"
if __name__ == '__main__':
    print(sale_data_month())