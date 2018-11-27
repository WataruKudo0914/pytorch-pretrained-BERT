import pandas as pd
import numpy as np


class Rating_Only(object):
    REV = 'reviewerID'
    PRODUCT = 'productID'
    RATING = 'rating'
    TIME = 'unixReviewTime'
    COLUMNS = [REV,PRODUCT,RATING,TIME]

    def load_data(self,file_path):
        ratings = pd.read_csv(file_path,header=None)
        ratings.columns = self.COLUMNS
        return ratings

    def scaler_per_reviewer(self,df):
        df_cp = df.copy()
        mean_rev = df_cp.groupby(self.REV)[self.RATING].mean()
        mean_rev.name = 'mean_rev'
        std_rev = df_cp.groupby(self.REV)[self.RATING].std()
        std_rev.name = 'std_rev'
        mean_std = pd.concat([mean_rev,std_rev],1)
        df_cp = pd.merge(df_cp,mean_std,left_on=self.REV,right_index=True)
        df_cp['rating-mean'] = df_cp[self.RATING] - df_cp['mean_rev']
        df_cp['rating-mean-bin'] = pd.cut(df_cp['rating-mean'],
                                        bins=[-5,-4,-3,-2,-1,0-1e-10,0,1,2,3,4,5],
                                        labels=range(-5,6)).astype(int)
        return df_cp

    def get_sorted_df(self,df,rating_column,lower=100,upper=float('inf')):
        selected = self.select_products(df,rating_column,lower,upper)
        merged_df = pd.merge(df,pd.DataFrame(selected),left_on=self.PRODUCT,
                                right_index=True)
        sorted_df = merged_df.sort_values([self.PRODUCT,self.TIME])
        sorted_df['increment'] = sorted_df.groupby(self.PRODUCT).cumcount()
        return sorted_df

    def make_RateAvgs(self,df,rating_column,limit_list=[5,10,20,50,100],
                        lower=100,upper=float('inf')):
        '''
            * making a DataFrame which has columns
                ["first 5 average rating", ... ,"ALL rating average"]
        '''
        sorted_df = self.get_sorted_df(df,rating_column,lower,upper)
        ret_df = pd.DataFrame(self.lim_means(sorted_df,sampling='all',
                                            rating_column=rating_column))
        for lim in limit_list:
            ret_df = pd.merge(ret_df,
                            pd.DataFrame(self.lim_means(sorted_df,rating_column,lim,'first')),
                            left_index=True,right_index=True)
        ret_df.columns = ['ALL'] + limit_list
        return ret_df

    def select_products(self,df,rating_column,lower=100,upper=float('inf')):
        '''
            * grouping by "productID"
            * getting (productID,review_counts) set which has more reviews than
                lower and less than upper
        '''
        product_count = df.groupby(self.PRODUCT).count()[rating_column]
        selected = product_count.loc[(lower<=product_count)&(product_count<=upper)]
        selected.name = 'counts'
        return selected

    def lim_means(self,df,rating_column,limit_num=10,sampling='first'):
        def masked_average(x):
            if sampling=='all':
                return np.average(x)
            elif sampling=='rand':
                return np.average(x,weights=np.random.permutation(len(x))<=limit_num)
            elif sampling=='first':
                return np.average(x,weights=np.arange(len(x))<limit_num)
            else:
                return np.average(x,weights=(np.arange(len(x))<limit_num)[::-1])
        return df.groupby(self.PRODUCT).agg(masked_average)[rating_column]

    def make_firstN_distribution(self,df,rating_column,limit_num=10):
        sorted_df = self.get_sorted_df(df,rating_column,lower=100,upper=float('inf'))
        firstN_df = sorted_df.loc[sorted_df['increment']<limit_num]
        dist = firstN_df.groupby([self.PRODUCT,rating_column]).count()[self.REV]
        dist = dist.unstack(1,fill_value=0)
        dist = dist / limit_num
        return dist

    def make_firstN_concats(self,df,rating_column,limit_num=10):
        sorted_df = self.get_sorted_df(df,rating_column)
        count_rev = df.groupby([self.REV,rating_column]).count()[self.PRODUCT]
        count_rev = count_rev.unstack(1,fill_value=0)
        dist_rev_np = count_rev.values/count_rev.values.sum(1).reshape(-1,1)
        dist_rev = pd.DataFrame(dist_rev_np,index=count_rev.index,
                                columns=count_rev.columns)
        merged = pd.merge(sorted_df.loc[sorted_df.increment<limit_num],dist_rev,
                            left_on=self.REV,right_index=True)
        merged['rate_dist'] = merged[[rating_column]+list(count_rev.columns)].apply(list,1)
        merged_gb = merged.groupby(self.PRODUCT)['rate_dist'].agg(sum)
        X_df = pd.DataFrame(np.vstack(merged_gb.values),index=merged_gb.index)
        return X_df

class Five_Core(Rating_Only):
    REV = 'reviewerID'
    HELPFUL = 'helpful'
    RATING = 'overall'
    TEXT = 'reviewText'
    PRODUCT = 'asin'
    TIME = 'unixReviewTime'
    HELPFUL_RATE = 'helpful_rate'
    VOTE_SUM = 'vote_sum'

    def load_data(self,file_path):
        return pd.read_json(file_path,lines=True)

    def get_helpfuls(self,df):
        h = df.copy().loc[df[self.HELPFUL].map(sum)>0]
        h[self.HELPFUL_RATE] = h[self.HELPFUL].map(lambda x:x[0]/x[1])
        h[self.VOTE_SUM] = h[self.HELPFUL].map(lambda x:x[1])
        return h

    def get_quintuple(self,df,lower):
        h = self.get_helpfuls(df)
        h = h.copy().loc[h[self.VOTE_SUM]>=lower]
        ret =  h.copy()[[self.REV,
                         self.PRODUCT,
                         self.TIME,
                         self.RATING,
                         self.TEXT,
                         self.HELPFUL_RATE]]
        return ret
