import pandas as pd
import math
from string import ascii_lowercase


def delta(col, ind1, ind2=None):
    if ind2 is None:
        return col[ind1] - col[ind1 - 1]
    return col[ind1] - col[ind2]


def clean_data(df):
    # delete rows that have repeated values that are the same as the one right above. 
    try:
        for index, row in df.iterrows():
            if row["x"] == df.iloc[index - 1]["x"] and row["y"] == df.iloc[index - 1]["y"]:
                df = df.drop(labels=index, axis=0)
    except:
        pass


def calculate_rubine(df):
    clean_data(df)

    x_col = df["x"]
    y_col = df["y"]
    t_col = df["time"]
    x_max = max(x_col)
    x_min = min(x_col)
    y_max = max(y_col)
    y_min = min(y_col)

    f1 = (x_col[2] - x_col[0]) / math.sqrt((y_col[2] - y_col[0]) ** 2 + (x_col[2] - x_col[0]) ** 2)
    f2 = (y_col[2] - y_col[0]) / math.sqrt((y_col[2] - y_col[0]) ** 2 + (x_col[2] - x_col[0]) ** 2)

    f3 = math.sqrt((y_max - y_min) ** 2 + (x_max - x_min) ** 2)
    f4 = math.atan2((y_max - y_min), (x_max - x_min))

    f5 = math.sqrt((x_col[df.shape[0] - 1] - x_col[0]) ** 2 + (y_col[df.shape[0] - 1] - y_col[0]) ** 2)
    f6 = (x_col[df.shape[0] - 1] - x_col[0]) / f5
    f7 = (y_col[df.shape[0] - 1] - y_col[0]) / f5

    f8 = 0
    for i in range(1, df.shape[0], 1):
        f8 += math.sqrt(delta(x_col, i) ** 2 + delta(y_col, i) ** 2)

    f9 = 0
    f10 = 0
    f11 = 0
    for i in range(2, df.shape[0], 1):
        nominator = (-delta(y_col, i) * delta(x_col, i - 1) + delta(y_col, i - 1) * delta(x_col, i))
        denominator = (delta(x_col, i) * delta(x_col, i - 1) + delta(y_col, i) * delta(y_col, i - 1))
        temp_angle = math.atan2(nominator, denominator)
        f9 += temp_angle
        f10 += abs(temp_angle)
        f11 += temp_angle ** 2

    f12 = 0
    for i in range(1, df.shape[0], 1):
        if delta(t_col, i) == 0:
            continue
        f12 = max(f12, (delta(x_col, i) ** 2 + delta(y_col, i) ** 2) / delta(t_col, i) ** 2)

    f13 = t_col[df.shape[0] - 1] - t_col[0]
    result = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13]
    return result


def generate1():  # run letter data
    results = []
    col = ["f01", "f02", "f03", "f04", "f05", "f06", "f07", "f08", "f09", "f10", "f11", "f12", "f13", "label"]
    for character in ascii_lowercase:
        for index in range(20):
            path = "data/letters-csv/" + character + "/" + character + "_" + str(index + 1) + ".csv"
            df = pd.read_csv(path)
            # print(path)
            result = calculate_rubine(df)
            result.insert(13, character)
            results.append(result)
    df_result = pd.DataFrame(results, columns=col)
    df_result.to_csv("features.csv", index=False)


def generate2():  # run sample data to test the correctness of code
    results = []
    col = ["sketch", "f01", "f02", "f03", "f04", "f05", "f06", "f07", "f08", "f09", "f10", "f11", "f12", "f13"]
    for index in range(8):
        path = "data/sample-csv/shape_" + str(index + 1) + ".csv"
        df = pd.read_csv(path)
        # print(path)
        result = calculate_rubine(df)
        result.insert(0, "shape_" + str(index + 1))
        results.append(result)
    df_result = pd.DataFrame(results, columns=col)
    df_result.to_csv("features.csv", index=False)


if __name__ == "__main__":
    generate1()
