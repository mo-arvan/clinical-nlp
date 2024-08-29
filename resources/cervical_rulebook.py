cervical_rulebook_definition = [
    {
        "category": "Satisfactory for evaluation",
        "pattern": r"((\bSatisfactory for evaluation(; transformation zone component present)?))",
        "metadata": {
            "group": "evaluation",
            "priority": 1,
        }
    },
    {
        "category": "Unsatisfactory for evaluation",
        "pattern": r"((Unsatisfactory for evaluation))",
        "metadata": {
            "group": "evaluation",
            "priority": 2,
        }
    },
    {
        "category": "Transformation zone component not identified",
        "pattern": r"(((\bSatisfactory for evaluation; )?transformation zone component not identified)|((\bSatisfactory for evaluation; )?endocervical/transformation component cannot\s+be assessed because of severe atrophy))",
        "metadata": {
            "group": "evaluation",
            "priority": 3,
        }
    },
    {
        "category": "Negative/Normal/NILM",
        "pattern": r"((Negative for intraepithelial lesion or malignancy)|(Negative for hyperplasia and malignancy)|(Negative for (definitive evidence of )?(high grade )?dysplasia( and malignancy)?))"
    },
    {
        "category": "Negative for Carcinoma",
        "literal": "Negative for carcinoma"
    },
    {
        "category": "Atypical Squamous Cells of undetermined Significance",
        "literal": "Atypical squamous cells of undetermined significance (ASC-US)"
    },
    {
        "category": "Atypical squamous cells cannot exclude HSIL atypical squamous cells (ASC-H)",
        "pattern": r"((Atypical squamous cells,? cannot exclude( a)? high(-| )grade squamous intraepithelial lesion( \(ASC-H\))?)|(ASC-H))"
    },
    {
        "category": "Low-grade squamous intraepithelial lesion (LSIL)",
        "pattern": r"((low(-| )(grade\s+)?squamous (intra)?epithelial lesion( \(LSIL\))?)|(LG?SIL))"
    },
    {
        "category": "High-grade squamous intraepithelial lesion (HSIL)",
        "pattern": r"((high(-| )(grade )?squamous (intra)?epithelial lesion( \(HSIL\))?(\. Cannot rule out squamous cell carcinoma)?)|(HG?SIL))"
    },
    {
        "category": "High-grade squamous intraepithelial lesion (HGSIL)",
        "pattern": r"((high(-| )(grade )?squamous (intra)?epithelial lesion( \(HGSIL\))?)|(ASCUS, cannot exclude HGSIL))"
    },
    {
        "category": "Squamous Cell Carcinoma",
        "literal": "Squamous Cell Carcinoma"
    },
    {
        "category": "Atypical endocervical cells",
        "literal": "Atypical endocervical cells"
    },
    {
        "category": "Atypical endometrial cells",
        "literal": "Atypical endometrial cells"
    },
    {
        "category": "Atypical glandular cells",
        "pattern": r"((Atypical glandular cells( \(AGC\))?)|(AGC))"
    },
    {
        "category": "Atypical endocervical cells (Favors neoplastic)",
        "literal": "Atypical endocervical cells, favor neoplastic"
    },
    {
        "category": "Atypical glandular cells (favor neoplastic)",
        "literal": "Atypical glandular cells, favor neoplastic"
    },
    {
        "category": "Endocervical Adenocarcinoma in Situ (AIS)",
        "pattern": r"(Endocervical )?Adenocarcinoma in(-| )situ( \(AIS\))?(( of)? cervix)?"
    },
    {
        "category": "Adenocarcinoma",
        "pattern": r"((Adenocarcinoma)|(carcinoma, adenocarcinoma))"
    },
    {
        "category": "Endocervical adenocarcinoma",
        "literal": "Endocervical adenocarcinoma"
    },
    {
        "category": "Endometrial adenocarcinoma",
        "literal": "endometrioid adenocarcinoma"
    },
    {
        "category": "Extrauterine adenocarcinoma",
        "literal": "Extrauterine adenocarcinoma"
    },
    {
        "category": "Adenocarcinoma NOS",
        "literal": "Adenocarcinoma NOS"
    },
    {
        "category": "Trichomonas Vaginalis",
        "pattern": r"((Organisms consistent with Trichomonas vaginalis)|(Trichomonas vaginalis))"
    },
    {
        "category": "Candida",
        "pattern": r"((Fungal organisms morphologically consistent with Candida spp.)|(Candida (species|spp(\.)?)))"
    },
    {
        "category": "Bacterial Vaginosis",
        "pattern": r"((Shift in flora suggestive of bacterial vaginosis)|(BACTERIAL VAGINOSIS HIGHLY SUGGESTIVE))"
    },
    {
        "category": "Actinomyces",
        "literal": "Bacteria morphologically consistent with Actinomyces spp"
    },
    {
        "category": "Herpes Simplex Virus",
        "literal": "Cellular changes consistent with herpes simplex virus"
    },
    {
        "category": "Cytomegalovirus",
        "literal": "Cellular changes consistent with cytomegalovirus"
    },
    {
        "category": "HPV Negative",
        "pattern": r"HPV\s+Negative"
    },
    {
        "category": "HPV 16 Negative",
        "pattern": r"HPV 16\s+Negative"
    },
    {
        "category": "HPV 18 Negative",
        "pattern": r"HPV 18\s+Negative"
    },
    {
        "category": "HPV Other Negative",
        "pattern": r"OTHER HIGH RISK HPV TYPES \*\s+Negative"
    },
    {
        "category": "HPV 18/45 Negative",
        "pattern": r"HPV 18/45\s+Negative"
    },
    {
        "category": "HPV Positive",
        "pattern": r"((HPV\s+Positive)|(POSITIVE FOR HIGH RISK HPV))"
    },
    {
        "category": "HPV 16 Positive",
        "pattern": r"HPV 16\s+Positive"
    },
    {
        "category": "HPV 18 Positive",
        "pattern": r"HPV 18\s+Positive"
    },
    {
        "category": "HPV Other positive",
        "pattern": r"OTHER HIGH RISK HPV TYPES \*\s+Positive"
    },
    {
        "category": "HPV 18/45 positive",
        "pattern": r"HPV 18/45\s+Positive"
    },
    {
        "category": "Benign",
        "literal": "benign"
    },
    {
        "category": "CIN 2-3",
        "pattern": r"(CIN (II|2)(-|/|, )(III|3))",
        "metadata": {
            "group": "CIN",
            "priority": 4,
        }
    },
    {
        "category": "CIN 1",
        "pattern": r"((mild dysplasia \(CIN I\))|(mild dysplasia)|(Low(-| )grade squamous (intra)?epithelial lesion)|(CIN (I|1)))",
        "metadata": {
            "group": "CIN",
            "priority": 1,
        }

    },
    {
        "category": "CIN 2",
        "pattern":
            r"((moderate dysplasia \(CIN II\))|(moderate dysplasia)|(cervical intraepithelial neoplasia \(CIN\) 2)|(high-grade squamous intraepithelial lesion)|(moderate grade dysplasia CIN II)|(HGSIL, high grade dysplasia CIN II)|(CIN (II|2)))",
        "metadata": {
            "group": "CIN",
            "priority": 2,
        }
    },
    {
        "category": "CIN3",
        "pattern": r"((severe dysplasia \(CIN III\))|(severe dysplasia)|(cervical intraepithelial neoplasia \(CIN\) 3)|(high(-| )grade squamous intraepithelial lesion)|(CIN III \(cervical intraepithelial neoplasia grade III\) with severe dysplasia)|(HGSIL, high grade dysplasia CIN II)|(CIN (III|3)))",
        "metadata": {
            "group": "CIN",
            "priority": 3,
        }
    },
    {
        "category": "Histologic HSIL (unspecified)",
        "pattern": r"((high(-| )grade squamous intraepithelial lesion)|(cannot rule out cancer))"
    },

    {
        "category": "Cancer NOS",
        "pattern": r"((carcinoma, not otherwise specified))"
    },
    {
        "category": "Small Cell Cancer",
        "literal": "carcinoma, small cell"
    },
    {
        "category": "Squamous Cell Cancer",
        "pattern": r"((carcinoma, squamous cell)|((Fragments of )?squamous carcinoma))"
    },
    {
        "category": "Endometrial Cancer",
        "pattern": r"((carcinoma, endometrial)|(endometrioid carcinoma))"
    },

    {
        "category": "Vaginal squamous intraepithelial neoplasia",
        "literal": "vaginal squamous intraepithelial neoplasia"
    },
]
