import pandas as pd  

#dc_genetic = load_obj("")

dc_iterative = load_obj("RESULT_ITERATIVE_50")
dc_genetic = load_obj("RESULTS_GENETIC")
dc_gradient = load_obj("RESULT_GRADIENT_E+0.33_C0.33")

df = pd.DataFrame()
for key in dc_genetic.keys():
    dcgenetic_time = dc_genetic[key]["TIME"]
    dcgenetic_score = dc_genetic[key]["SCORE"]

    dciterative_time = dc_iterative[key]["TIME"]
    dciterative_score = dc_iterative[key]["SCORE"] 

    dcgradient_time = dc_gradient[key]["TIME"]
    dcgradient_score = dc_gradient[key]["SCORE"]
    
    df.loc[cont, "MODEL"] = "GENETIC"
    df.loc[cont,"Value"] = dcgenetic_score
    df.loc[cont, "type"]  = "SCORE"
    df.loc[cont, "key"]   = key
    
    df.loc[cont+1, "MODEL"] = "GENETIC"
    df.loc[cont+1, "Value"] = dcgenetic_time
    df.loc[cont+1, "type"]  = "TIME" 
    df.loc[cont+1, "key"] = key
    
    df.loc[cont+2, "MODEL"] = "ITERATIVE"       
    df.loc[cont+2,"Value"]  = dciterative_score 
    df.loc[cont+2, "type"]  = "SCORE"
    df.loc[cont+2, "key"]   = key

    df.loc[cont+3, "MODEL"] = "ITERATIVE"       
    df.loc[cont+3, "Value"] = dciterative_time
    df.loc[cont+3, "type"]  = "TIME"  
    df.loc[cont+3, "key"] = key 


    df.loc[cont+4, "MODEL"] = "GRADIENT"       
    df.loc[cont+4,"Value"]  = dcgradient_score 
    df.loc[cont+4, "type"]  = "SCORE"
    df.loc[cont+4, "key"]   = key

    df.loc[cont+5, "MODEL"] = "GRADIENT"       
    df.loc[cont+5, "Value"] = dcgradient_time
    df.loc[cont+5, "type"]  = "TIME"  
    df.loc[cont+5, "key"] = key 

    cont +=6
