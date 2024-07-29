import dataclasses

cervical_rulebook_definition = [
    {
        "category": "Satisfactory for evaluation",
        "pattern": "Satisfactory for evaluation(; transformation zone component present)?"
    },
    {
        "category": "Unsatisfactory for evaluation",
        "literal": "Unsatisfactory for evaluation"

    },
    {
        #     transformation zone component not identified
        "category": "Transformation zone component not identified",
        "pattern": r"(Satisfactory for evaluation; )?transformation zone component not identified"
    },
    {
        "category": "Negative/Normal/NILM",
        "pattern": r"((Negative for intraepithelial lesion or malignancy)|(Negative for hyperplasia and malignancy)|(Negative for dysplasia and malignancy))"
        # Negative for hyperplasia and malignancy
        # Negative for dysplasia and malignancy

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
        "pattern": r"Atypical squamous cells,? cannot exclude( a)? high(-| )grade squamous intraepithelial lesion \(ASC-H\)"
        #     Atypical squamous cells cannot exclude a high-grade squamous intraepithelial lesion (ASC-H)
        # Atypical squamous cells, cannot exclude a high-grade squamous intraepithelial lesion (ASC-H)
    },
    {
        "category": "Low-grade squamous intraepithelial lesion (LSIL)",
        "pattern": r"Low(-| )grade squamous intraepithelial lesion( \(LSIL\))?"

        #     low-grade squamous intraepithelial lesion (LSIL).
        # low-grade squamous intraepithelial lesion
    },
    {
        "category": "High-grade squamous intraepithelial lesion (HSIL)",
        "pattern": r"high(-| )grade squamous intraepithelial lesion( \(HSIL\))?(\. Cannot rule out squamous cell carcinoma)?"
        #     high-grade squamous intraepithelial lesion (ASC-H)
        #     high grade squamous intraepithelial lesion (HSIL)
    },
    {
        "category": "High-grade squamous intraepithelial lesion (HGSIL)",
        "pattern": r"high(-| )grade squamous intraepithelial lesion \(HGSIL\)"
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
        "literal": "Atypical glandular cells"
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
    # {
    #     "category": "AIS",
    #     "pattern": r"((Adenocarcinoma in(-| )situ( \(AIS\))?)(( of)? cervix)?)"
    #     #     Adenocarcinoma in-situ
    #     #  Adenocarcinoma in-situ (AIS)
    # },

    {
        "category": "Adenocarcinoma",
        "literal": "Adenocarcinoma"
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
        "literal": "Organisms consistent with Trichomonas vaginalis"
    },
    {
        "category": "Candida",
        "literal": "Fungal organisms morphologically consistent with Candida spp."
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
        #     OTHER HIGH RISK HPV TYPES *
        #     (*INCLUDES TYPES 31, 33, 35, 39, 45, 51, 52, 56, 58, 59, 66, 68)
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
        "category": "CIN 1",
        "pattern": r"((mild dysplasia \(CIN I\))|(mild dysplasia)|(Low(-| )grade squamous (intra)?epithelial lesion)|CIN I)"
        #     low grade squamous epithelial lesion
    },
    {
        "category": "CIN 2",
        "pattern":
            r"((moderate dysplasia \(CIN II\))|(moderate dysplasia)|(cervical intraepithelial neoplasia \(CIN\) 2)|(high-grade squamous intraepithelial lesion)|(moderate grade dysplasia CIN II)|(HGSIL, high grade dysplasia CIN II))"
        #     CIN II
        # HGSIL, high grade dysplasia CIN II
        # moderate grade dysplasia CIN II
    },
    {
        "category": "CIN3",
        "pattern": r"((severe dysplasia \(CIN III\))|(severe dysplasia)|(cervical intraepithelial neoplasia \(CIN\) 3)|(high(-| )grade squamous intraepithelial lesion)|(CIN III \(cervical intraepithelial neoplasia grade III\) with severe dysplasia)|(HGSIL, high grade dysplasia CIN II))"
        #     high grade squamous intraepithelial lesion
    },
    {
        "category": "CIN 2-3",
        "pattern": r"(CIN II(-|/)III)"
    },
    {
        "category": "Histologic HSIL (unspecified)",
        "pattern": r"((high(-| )grade squamous intraepithelial lesion)|(cannot rule out cancer))"
    },

    {
        "category": "Cancer NOS",
        "literal": "carcinoma, not otherwise specified"
    },
    {
        "category": "Small Cell Cancer",
        "literal": "carcinoma, small cell"
    },
    {
        "category": "Squamous Cell Cancer",
        "literal": "carcinoma, squamous cell"
    },
    {
        "category": "Endometrial Cancer",
        "literal": "carcinoma, endometrial"
    },
    {
        "category": "Adenocarcinoma",
        "literal": "carcinoma, adenocarcinoma"
    },
    {
        "category": "Vaginal squamous intraepithelial neoplasia",
        "literal": "vaginal squamous intraepithelial neoplasia"
    },
    # {
    #     "category": "",
    #     "literal": [
    #         ""
    #     ]
    # },
]
