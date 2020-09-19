import pandas as pd
import os
import matplotlib.pyplot as plt

# Both return a pandas Dataframe object
def load_freedom():
    csv_path = "/home/christian/Documents/Database/efw_cc.csv"
    return pd.read_csv(csv_path)

def load_birth():
    csv_path = "/home/christian/Documents/Database/MortalityFertilityIncome.csv"
    return pd.read_csv(csv_path)

freedom_features = ["year", "countries", "ISO_code", "EF", "rank"]
birth_features = ["country", "code"]

def build_birth_features():
    for i in range(0,10):
        birth_features.append("f197"+ str(i))
        birth_features.append("f198"+ str(i))
        birth_features.append("f199"+ str(i))
        birth_features.append("f200"+ str(i))
        birth_features.append("f201"+ str(i))

        birth_features.append("i197"+ str(i))
        birth_features.append("i198"+ str(i))
        birth_features.append("i199"+ str(i))
        birth_features.append("i200"+ str(i))
        birth_features.append("i201"+ str(i))


def main():
    freedom = load_freedom()
    print(freedom.head())

    # Drop any column that is not a wanted feature
    for column, values in freedom.iteritems():
        # print("Column:", column)
        if column not in freedom_features:
            # print("Not", column)
            freedom.drop(columns = [column], inplace=True)

    print("////////////////////////////////////////////////////////")
    freedom.dropna(inplace= True)
    print(freedom.head())

    birth = load_birth()
    build_birth_features()
    # print(birth_features)
    # Drop any column not in a wanted feature
    for column, values in birth.iteritems():
        # print("Column:", column)
        if column not in birth_features:
            # print("Not", column)
            birth.drop(columns = [column], inplace=True)
    birth.dropna(inplace=True)
    birth.replace(',', '', regex = True, inplace=True)
    print(birth.head())

    freedom.to_csv('freedom.csv', index=False)
    birth.to_csv('birth.csv', index = False)

if __name__ == "__main__":
    main()