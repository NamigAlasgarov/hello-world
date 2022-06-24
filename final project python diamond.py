#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import stemgraphic
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import pylab
import numpy as np
import missingno as msno


# In[2]:


df = pd.read_csv(r'C:\Users\Namiq\Desktop\diamonds.csv')
df = df.drop(df.columns[[0]], axis=1)
df.head(20)


# In[3]:


cut_num = df['cut'].value_counts()
cut_num


# In[4]:


df.shape


# ## Data check and cleaning

# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


msno.matrix(df)


# Burada datanı yüklədikdən sonra Miisingno Library vasitəsilə məlumat boşluqlarının olmadığına əmin ola bilərik.
# daha sonra isə describe ilə sütunlar haqqında ümumi məlumat alaq

# In[8]:


df.describe()


# Gördüyünüz kimi burada x y və z sütunları üçün min dəyər 0 olaraq göstərilib. 
# yəni hər hansı bir almazın ölçüləri sıfır ola bilməyəcəyinə görəbunları silməy ən optimal seçimdir.
# 
# PS: bunu internetdə bu datanın edilmiş analizində gördüm və mənədə məntiqli gəldiyi üçün tətbiq etmək istədim. 

# In[9]:


df.loc[(df['x']==0) | (df['y']==0) | (df['z']==0)]


# In[10]:


len(df[(df['x']==0) | (df['y']==0) | (df['z']==0)])


# In[11]:


df = df[(df[['x','y','z']] != 0).all(axis=1)]


# In[12]:


df.loc[(df['x']==0) | (df['y']==0) | (df['z']==0)]


# In[13]:


df.describe()


# ## Distribution of columns and correlation

# x y və z sütunlarının hər hansı birində sıfır dəyər olan sətiri sildikdən sonra describe ilə bir daha yoxlayırıq və 
# gördüyünüz kimi artıq qeyd edilən sütunlarda min dəyər sıfır deyil.

# In[172]:


corr = df.corr()
sns.heatmap(data=corr, square=True , annot=True, cbar=True)


# yuxarıda isə seaborn heatmap vasitəsilə datanın correlyasiya vizualını var, burada bütün sütunların bir birindən asılılığını 
# -1 və 1 aralığında görmək olar, əgər  asılılıq mənfidirsə tərs asılılıq əksi halda düz asılılıq mövcuddur deməkdir. 
# 
# visualdan göründüyü kimi carat ilə price, x,y,z sütunlarının yüksət düz asılılığı var yəni carat nə qədər böyükdürsə 
# price da o qədər yüksəkdir. Depth və table arasında ən böyük mənfi əlaqənin də mövcud olduğunu görə bilərik. 
# 
# Ümumiyyətlə table və depth sütunlarının digərlərinə nisbətən correliyasiyaları sıfıra çox yaxın mənfi və ya müsbət rəqəmlərdir.
# 
# Bunun səbəbi table və depth ölçülərinin bir birindən asılı olması və onların özlərinəməxsus ideal ölçü dərəcələrinin olmasıdır. 
# yəni onların bir biri ilə ən ideal dərəcədə olması çox önəmlidir nəinki ayrı ayrılıqda price və ya carat sütunu ilə 
# correliyasiya dərəcəsi yüksək olsun.
# 
# bu qayda ilə bütün sütunları bir biri ilə correliyasiya dərəcələrini müqayisə etmək olar.
# 
# 

# In[171]:


hist = df.hist(bins=10, figsize=(15, 10))


# histogram vasitəsilə rəqəmlərdən ibarət bütün sütunların paylanmasını visual olaraq görmək mümkündür.
# ilkin baxışda caratın daha çox sıfır və iki arasında paylandığını gprə bilərik və ya price sütununda  daha çox 2500də diamond sayı çoxdur və getdikcə sürətlə azalır. ancaq bar char vasitəsilə visualizasiyə çoxda yaxşı olmaya bilər, bunuları seaborn kdeplot vasitəsilə daha aydın visuallaşdıra bilərik.

# In[86]:


sns.kdeplot(df['carat'], shade=True , color='r')
plt.rcParams["figure.figsize"] = [8, 5]


# In[85]:


sns.kdeplot(df['price'], shade=True , color='r')
plt.rcParams["figure.figsize"] = [8, 5]


# In[174]:


sns.kdeplot(df['x'], shade=True , color='r')
plt.rcParams["figure.figsize"] = [8, 5]


# In[178]:


diamonds=df['carat']
stemgraphic.stem_graphic(diamonds, scale=0.5)
plt.show()


# In[105]:


diamonds=df['table']
stemgraphic.stem_graphic(diamonds, scale=1)
plt.show()


# ## One sample t-test

# aşağıda biz hər bir sütunun yuxarıda olduğu kimi dağılımlarını görə bilərk və bundan əlavə normal distribution olub olmadığını
# yoxlaya bilərik. aşağıdakılardan göründüyü kimi heç bir stunun paylanması normal deyil.

# In[ ]:


price= df['price']


# In[146]:


sns.histplot(price)
plt.show()


# In[147]:


sns.distplot(price)
plt.show()


# In[148]:


sns.distplot(price, fit=stats.norm)
plt.show()


# In[149]:


stats.probplot(price, dist='norm', plot=pylab)
pylab.show()


# In[150]:


carat = df['carat']


# In[152]:


sns.distplot(carat, fit=stats.norm)
plt.show()


# In[153]:


x= df['x']


# In[154]:


sns.distplot(x, fit=stats.norm)
plt.show()


# In[156]:


depth = df['depth']
sns.distplot(depth, fit=stats.norm)
plt.show()


# In[159]:


print(stats.shapiro(depth))


# In[161]:


print(stats.shapiro(price))


# In[162]:


print(stats.shapiro(carat))


# In[180]:


table = df['table']
print(stats.shapiro(table))


# In[183]:


print(len(df))


# Shapiro testinin müxtəlif sütunlara tətbiqi  nəticəsində pvalue 0.05 dən kiçik olduğunu görüb bir daha heç bir sütunun 
# paylanması normal olmadığını deyə bikrik.
# ancaq datamızın sayı 120 dən çox olduğu üçün t test tətbiq edə bilərik.
# 
# 
# ilk öncə price sütunu üzrə t testini tətbiq etməyə çalışaq

# In[163]:


sns.boxplot(price)
plt.show()


# burada ilk öncə null hypothesis i 5000ə bərabər olub olmadığını yoxlayaq.
# 
# H0 (null hypothesis): mu=5000                             
# 
# H1 (alternative hypotesis): mu!=5000

# In[195]:


t_statistic, p_value=stats.ttest_1samp(price, 5000, alternative='two-sided')
print(f't statistic: {t_statistic}')

t_c=stats.t.ppf(q=1-.05/2,df=len(price)-1)
print(f't critic: {t_c}')


# nəticədə t statistik dəyər t critic dəyərdən kiçik olduğu üçün H0 reject edilir, yəni mu 5000ə bərabər deyil.
# 
# bu haldsa mu 5000dən böyük və ya kiçik ola bilər.
# növbəti testlə yoxlayaq.
# 
# H0 (null hypothesis): mu5000
# 
# H1 (alternative hypotesis): mu<5000

# In[191]:


t_statistic, p_value=stats.ttest_1samp(price, 5000, alternative='less')
print(f't statistic: {t_statistic}')

t_c=stats.t.ppf(q=.05,df=len(price)-1)
print(f't critic: {t_c}')


# burada da t statistic t critic dəyərdən kiçik olduğu üçün H0 reject edilir, yəni alternativ hipotezə görə mu 5000dən kiçikdir. 

# son olaraq bu datada pvalue bütün sütunlar üzrə o.o5 dən kiçikdir və sütunların paylanması normal deyil 
# amma sətir sayı 120dən böyük olduğu üçün t test tətbiq etdik. Bu səbəbdən də Wilcoxon Rank Sum test tətbiq etmədik.
# 

# In[193]:


confidence_interval = stats.t.interval( alpha=0.95, df=len(price)-1, loc=np.mean(price), scale=stats.sem(price) )
print(confidence_interval)


# ## Scatter plot-correlation

# In[9]:


plt.plot(df.price, df.carat,'o',markersize=1, color='blue')
plt.xlabel('price')
plt.ylabel('carat')


# 1) sütunların bir biri ilə correlyasiya dərəcələrini scatter plot vasitəsilə də ölçmək olar. price və carat sütunlarının 
# qrafikinə əsasən qiymət artsada almazın carat dəyəri aşağıdan başlayır və daha çox 1 və 3 carat arasında olur. 
# 
# 
# 2) aşağıdakı ikinci şəkildə isə  ən yüksək correlyasiya carat və x sütunları arasındadır 0.98, yəni almazın x ölçüsü və 
# carat dəyəri arasında müsbət tənasüblük var, amma bu müsbət tənasüblük x ölçüsü üçün getdikcə azalır.
# 
# 
# 3) sonda isə price və depth sütunları arasında demək olar ki heç bir əlaqə olmadığını görə bilərik.

# In[10]:


plt.plot(df.carat, df.x,'o',markersize=1, color='red')
plt.xlabel('carat')
plt.ylabel('x')


# In[16]:


plt.plot(df.price, df.depth,'o',markersize=1, color='green')
plt.xlabel('price')
plt.ylabel('depth')


# ## Two sample t-test

# In[11]:


price = df['price']
sns.distplot(price, fit=stats.norm)
plt.show()


# In[12]:


carat = df['carat']
sns.distplot(carat, fit=stats.norm)
plt.show()


# In[18]:


price = df['price']
print(stats.shapiro(price))


# In[19]:


carat = df['carat']
print(stats.shapiro(carat))


# In[ ]:


burada hər iki columnda da shapiro test nəticələrinə görə pvalue 0.05dən kiçik olduğu üçün normal paylanma olmasada 
sətirlərin sayı 120dən çox olduğu  üçün t test edə bilərik.


# H0 (null hypothesis): mu1-mu2=0
#     
# H1 (alternative hypotesis): mu1-m2!=0

# In[24]:


print(stats.ttest_ind(carat, price, alternative='two-sided', equal_var=True))

print(stats.t.ppf(q=1-.05/2,df=len(carat)-1))


# pvalue 0.05 dən kiçik olduğu üçün H0 rejet edilir yəni iki qrupun meanləri bir birindən fərqlidir.
# 
# H0 (null hypothesis): mu1-mu2=0
# 
# 
# H1 (alternative hypotesis): mu1-m2<0
# 

# In[23]:


print(stats.ttest_ind(carat, price, alternative='less'))

print(stats.t.ppf(q=.05,df=len(carat)-1))


# alınan son nəticəyə görə H0 reject olunur və price mean ortalaması caratın mean ortalamasından böyük olduğunu göstərir.

# In[ ]:




