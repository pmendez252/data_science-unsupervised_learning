function [data] = handle_missing(data)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Replace missing values by attribute mean %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create matrix with attribute means, where missing values exists for data
means = bsxfun(@times, isnan(data), nanmean(data));

% Replace NaN by the means
data(isnan(data)) = 0;
data = data + means;




end