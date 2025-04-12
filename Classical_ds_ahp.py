
import pandas as pd
import numpy as np
import numpy.linalg as la
from numpy import intersect1d, union1d
from numpy import argmax, argsort
from math import sqrt



"""
Fonction qui prend en entrée la liste de favorabilités et l'ensemble des alternatives
puis regroupe les aternatives ayant la même favorabilité dans un dictionnaire
"""
def focal_elements(favorability: list, alternatives: list):
    fTemp = favorability[:]
    n = len(fTemp)
    focal_element = []
    f_dict = dict()

    for i in range(n):

        if fTemp[i]  != 0:
            focal_element.append(alternatives[i])
        for j in range(i+1, n):
            if fTemp[i] == fTemp[j] and fTemp[i] != 0:
                focal_element.append(alternatives[j])
                fTemp[j] = 0
        if fTemp[i]  != 0:
            f_dict[(fTemp[i])] = focal_element
        focal_element = []

    return f_dict

def DM_BOEs(decision_data, criteria_list):
  DM_number = decision_data[criteria_list[0]].shape[1]
  alternatives = np.arange(1,45001)
  DM_judgements = []
  for i in range(DM_number):
    criteria_BOE = {}
    for criterion in criteria_list:
      df = decision_data[criterion].copy()
      dm_data = df[:,i]
      filter = np.isnan(dm_data) == False
      temp = alternatives[filter]
      favorability = dm_data[filter]
      criteria_BOE[criterion] = focal_elements(favorability, temp)
      favorability = []
      DM_judgements.append(criteria_BOE)

  return DM_judgements

"""
Fonction qui calcule la valeur de masse d'un élément focal
"""
def mass_value(criterion_BOE: dict, criterion_priority: float, scale_value: int):
    sum_priorities = 0
    for k, v in criterion_BOE.items():
        sum_priorities += k

    mass = (scale_value * criterion_priority) / (sum_priorities * criterion_priority + sqrt(len(criterion_BOE)))
    return mass

"""
Fonction qui calcule le niveau d'ignorance d'un BOE
"""
def ignorance_level(criterion_BOE: dict, criterion_priority: float):
    sum_priorities = 0
    for k, v in criterion_BOE.items():
        sum_priorities += k

    mass = sqrt(len(criterion_BOE)) / (sum_priorities * criterion_priority + sqrt(len(criterion_BOE)))
    return mass


"""
Fonction qui effectue la combinaison des jugements des décideurs
"""
def dempster_combination(boe_tab:list):

    while len(boe_tab) > 1:
        inter=intersect_BOE(boe_tab[0], boe_tab[1])
        boe_tab.pop(0)
        boe_tab[0] = combined_BOE(inter[0],inter[1], inter[2])
    return boe_tab[0]


"""
Fonction qui retourne la somme des valeurs de masse de tous les couples
d'éléments focaux dont leur intersection est nulle
"""
def inter_null_sum(inter_nul: list):
    if type(inter_nul) is list:
        sum = 0
        if inter_nul != []:
            for i in range(len(inter_nul)):
                sum += inter_nul[i][0]*inter_nul[i][1]
        return sum

"""
Fonction qui effectue la combinaison de deux BOE
"""
def combined_BOE(inter_null: list, not_null_inter: list, inter_result: list):
    inter_nul_sum = inter_null_sum(inter_null)
    not_null_tab = not_null_inter
    inter_tab = inter_result
    combined_boe = {}
    j=1
    while inter_tab != []:
        combined_mass_value = not_null_tab[0][0]*not_null_tab[0][1]
        i=1
        while i < len(inter_tab):
            equal = True
            for k in range(len(inter_tab[i])) :
                if inter_tab[i][k] not in inter_tab[0]:
                    equal = False
                    break
            if equal == True:
                combined_mass_value += not_null_tab[i][0]*not_null_tab[i][1]
                inter_tab.pop(i)
                not_null_tab.pop(i)
                continue
            i += 1
        combined_mass_value /= (1-inter_nul_sum)
        combined_boe[(combined_mass_value,j)] = inter_tab[0]
        inter_tab.pop(0)
        not_null_tab.pop(0)
        j += 1
    return combined_boe

"""
Fonction qui effectue l'intersection de deux BOE
"""
def intersect_BOE(BOE1: dict, BOE2: dict):
    inter_null = []
    not_null_inter = []
    inter_result = []
    intersect = []
    for m1,boe1 in BOE1.items():
        for m2,boe2 in  BOE2.items():
            print(boe1, boe2)
            intersect = list(intersect1d(boe1, boe2))
            if intersect == []:
                inter_null.append((m1[0], m2[0]))
            else:
                not_null_inter.append((m1[0], m2[0]))
                inter_result.append(intersect)
    return inter_null, not_null_inter, inter_result


"""
Fonction qui calcule la croyance d'un BOE
"""
def boe_belief(boe:dict):
    tab_mass = []
    tab_foc_element = []
    belief_foc_el = {}
    for m,v in boe.items():
        tab_mass.append(m)
        tab_foc_element.append(v)
    for i in range(len(tab_foc_element)):
        belief_value = tab_mass[i][0]
        for j in range(len(tab_foc_element)):
            if j != i:
                if list(intersect1d(tab_foc_element[j],tab_foc_element[i])) == tab_foc_element[j]:
                    belief_value += tab_mass[j][0]
        belief_foc_el[(belief_value,i)] = tab_foc_element[i]
    return belief_foc_el


"""
Fonction qui calcule la croyance des BOE de tous les décideurs
"""
def DMs_boe_belief(tab_foc:list):
    DMs_boe_tab = []
    for i in range(len(tab_foc)):
        DMs_boe_tab.append(boe_belief(tab_foc[i]))
    return DMs_boe_tab

"""
Fonction qui calcule la plausibilité d'un BOE
"""
def boe_plausibility(boe:dict):
    tab_mass = []
    tab_foc_el = []
    plausi_foc_el = {}
    for m,v in boe.items():
        tab_mass.append(m)
        tab_foc_el.append(v)
    for i in range(len(tab_foc_el)):
        plausi_value = tab_mass[i][0]
        for j in range(len(tab_foc_el)):
            if j != i:
                if list(intersect1d(tab_foc_el[j],tab_foc_el[i])) != []:
                    plausi_value += tab_mass[j][0]
        plausi_foc_el[(plausi_value,i)] = tab_foc_el[i]
    return plausi_foc_el

"""
Fonction qui calcule la plausibilité des BOE de tous les décideurs

"""
def DMs_boe_plausibility(tab_foc:list):
    DMs_boe_tab = []
    for i in range(len(tab_foc)):
        DMs_boe_tab.append(boe_plausibility(tab_foc[i]))
    return DMs_boe_tab