# stolen two coil EDFA model from GNPy 
import numpy as np
import math
def lin2db(x):
    return 10*np.log10(x)

def db2lin(x):
    return 10**(x/10)

def nf_model(gain_min, gain_max, nf_min, nf_max):

# NF estimation model based on nf_min and nf_max
    # delta_p:  max power dB difference between first and second stage coils
    # dB g1a:   first stage gain - internal VOA attenuation
    # nf1, nf2: first and second stage coils
    #           calculated by solving nf_{min,max} = nf1 + nf2 / g1a{min,max}
    delta_p = 5
    g1a_min = gain_min - (gain_max - gain_min) - delta_p
    g1a_max = gain_max - delta_p
    nf2 = lin2db((db2lin(nf_min) - db2lin(nf_max)) /
                 (1/db2lin(g1a_max) - 1/db2lin(g1a_min)))
    nf1 = lin2db(db2lin(nf_min) - db2lin(nf2)/db2lin(g1a_max))
    
    if nf1 < 4:
        print("First coil value too low")
        #raise EquipmentConfigError(f'First coil value too low {nf1} for amplifier {type_variety}')

    # Check 1 dB < delta_p < 6 dB to ensure nf_min and nf_max values make sense.
    # There shouldn't be high nf differences between the two coils:
    #    nf2 should be nf1 + 0.3 < nf2 < nf1 + 2
    # If not, recompute and check delta_p
    if not nf1 + 0.3 < nf2 < nf1 + 2:
        nf2 = np.clip(nf2, nf1 + 0.3, nf1 + 2)
        g1a_max = lin2db(db2lin(nf2) / (db2lin(nf_min) - db2lin(nf1)))
        delta_p = gain_max - g1a_max
        g1a_min = gain_min - (gain_max-gain_min) - delta_p
        if not 1 < delta_p < 11:
            #raise EquipmentConfigError(f'Computed \N{greek capital letter delta}P invalid \
            #    \n 1st coil vs 2nd coil calculated DeltaP {delta_p:.2f} for \
            #    \n amplifier {type_variety} is not valid: revise inputs \
            #    \n calculated 1st coil NF = {nf1:.2f}, 2nd coil NF = {nf2:.2f}')
            print("Computed \N{greek capital letter delta}P invalid")
    # Check calculated values for nf1 and nf2
    calc_nf_min = lin2db(db2lin(nf1) + db2lin(nf2)/db2lin(g1a_max))
    if not math.isclose(nf_min, calc_nf_min, abs_tol=0.01):
        #raise EquipmentConfigError(f'nf_min does not match calc_nf_min, {nf_min} vs {calc_nf_min} for amp {type_variety}')
        print("nf_min does not match calc_nf_min")
    calc_nf_max = lin2db(db2lin(nf1) + db2lin(nf2)/db2lin(g1a_min))
    if not math.isclose(nf_max, calc_nf_max, abs_tol=0.01):
        #raise EquipmentConfigError(f'nf_max does not match calc_nf_max, {nf_max} vs {calc_nf_max} for amp {type_variety}')
        print("nf_max does not match calc_nf_max")
    
    return nf1, nf2, delta_p, g1a_max, g1a_min




