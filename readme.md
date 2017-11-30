## Postup riešenia
Celá detekcia bola prezatiaľ realizovaná iba na malej sade obrázkov, myslím, že pamäť RAM by nemala byť problémom, budem to spracovávať iba po jednotlivých sadách, kde jedna sada má $\pm$ 150MB, v pamäti to bude do 500MB potom.

Postup je následovný:

1. Detekcia bodov, ktoré by mohli odpovedať výboju, problém je kvantový šum a prekrývajúce sa výboje.
2. Matematická erózia, odstránim tak kvantový šum čiastočne, ktorý predpokladám, že je bodový.
3. Matematická dilatácia, spojí vzdialenejšie body od seba. (Mám zarátať aj tvar štrukturálneho elemmentu, kaďže predpokladáme, že výboj je vyšší a iba málo široký?).
4. Výber iba naozaj jasných oblastí, ktoré sú "semienkami" výsledných segmentov.
5. Výber masky (všetky body, ktoré majú byť priradené k oblastiam).
6. Watershed s parametrami:
    * Obrázok originálny, naprahovaný.
    * Semienka, ktoré sú očíslované v určitom poradí.
    * Maska s bodmi, ktoré budeme segmentovať (segmentácia celého obrázku nemá zmysel)
7. Výsledkom je obrázok s oblasťami číslovanými 0 - pozadie, hodnoty $\left[1,n\right]$ s jednotlivými segmentovanými oblasťami.



## Možné vylepšenia
Preteraz ma napadajú niektoré z vylepšení:

* Potlačenie "semienok", ktoré majú pod určitou veľkosťou. Problém môže byť, že následne nebudeme schopní detekovať niektoré menšie výboje teoreticky.
* Odstránenie z finálneho obrázku malé detekované objekty, rovnaký problém ako v predošlom príklade.
* Ako nastaviť hodnoty prahu?
* Iná metóda ako watershed?



## TO-DO list
   - [ ] Otestovať funkčnosť na výbojoch v druhej polperióde.
   - [x] Detekcia jednotlivých výbojov aspoň na jednom obrázku.
   - [ ] Načítanie obrázkov v rade.


# Predspracovanie obrazov
V prvom rade bude realizované predspracovanie obrazov, za účelom detekcie jednotlivých pixelov.
