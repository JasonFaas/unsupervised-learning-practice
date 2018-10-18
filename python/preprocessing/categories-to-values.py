import pandas as pd
from sklearn.preprocessing import scale

df = pd.read_csv('vgsales.csv')
print(df.info())
print(df.describe())
# print(df[df.isnull().any(axis=1)])
# exit(0)
# df = df.dropna()

# df_origin = pd.get_dummies(df, columns=['Platform', 'Genre', 'Publisher'])
df_origin = pd.get_dummies(df, columns=['Platform', 'Genre'])
# df_origin = pd.get_dummies(df, columns=['Genre'])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
x = df_origin[['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales',
       'Other_Sales', 'Global_Sales', 'Platform_2600', 'Platform_3DO',
       'Platform_3DS', 'Platform_DC', 'Platform_DS', 'Platform_GB',
       'Platform_GBA', 'Platform_GC', 'Platform_GEN', 'Platform_GG',
       'Platform_N64', 'Platform_NES', 'Platform_NG', 'Platform_PC',
       'Platform_PCFX', 'Platform_PS', 'Platform_PS2', 'Platform_PS3',
       'Platform_PS4', 'Platform_PSP', 'Platform_PSV', 'Platform_SAT',
       'Platform_SCD', 'Platform_SNES', 'Platform_TG16', 'Platform_WS',
       'Platform_Wii', 'Platform_WiiU', 'Platform_X360', 'Platform_XB',
       'Platform_XOne', 'Genre_Action',
       'Genre_Adventure', 'Genre_Fighting', 'Genre_Misc', 'Genre_Platform',
       'Genre_Puzzle', 'Genre_Racing', 'Genre_Role-Playing', 'Genre_Shooter',
       'Genre_Simulation', 'Genre_Sports', 'Genre_Strategy']]
y = df_origin['Rank'].values
x = x.values

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(x)
x = imp.transform(x)

x = scale(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=91)

ridge = Ridge(alpha=0.5, normalize=True).fit(x_train, y_train)

print(ridge.score(x_test, y_test))

