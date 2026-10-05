# Camiq SigLIP Evaluation Report: QVHighlights Benchmark

**Date:** 2026-10-05 16:01:16  
**Model Architecture:** ViT-B-16-SigLIP (HF Tokenizer context length 64)  
**Evaluated Dataset:** `highlight_train_release.jsonl`  
**Evaluated Samples:** 100  
**Total Wall Time:** 9282.56s  

---

## 1. Summary of Performance Metrics (Research Benchmark)

| Metric | Score | Definition |
| :--- | :--- | :--- |
| **Hit@1** | **72.00%** | Top-1 retrieved frame lands in ground truth window |
| **Hit@5** | **93.00%** | Top-5 retrieved frames include ground truth window |
| **Hit@10** | **99.00%** | Top-10 retrieved frames include ground truth window |
| **R@1 (tIoU ≥ 0.5)** | **1.00%** | Top-1 moment overlap tIoU ≥ 0.5 |
| **R@5 (tIoU ≥ 0.5)** | **3.00%** | Top-5 moment overlap tIoU ≥ 0.5 |
| **R@1 (tIoU ≥ 0.7)** | **1.00%** | Strict temporal overlap tIoU ≥ 0.7 |
| **R@5 (tIoU ≥ 0.7)** | **2.00%** | Strict temporal overlap tIoU ≥ 0.7 |
| **MRR** | **0.8171** | Mean Reciprocal Rank across queries |
| **mAP** | **0.7125** | Mean Average Precision across temporal windows |
| **Saliency Spearman $\rho$** | **-0.1846** | Rank correlation with human saliency annotations |

---

## 2. Pipeline Efficiency & Frame Filtering Analysis

| Efficiency Metric | Average Value per Video |
| :--- | :--- |
| **Indexing Latency** | 20.82 s |
| **Retained Frames** | 49.4 frames |
| **Static Frames Pruned** | 22.7 frames |
| **Semantic Duplicates Pruned** | 75.3 frames |

---

## 3. Detailed Per-Query Results

| QID | Video ID | Query | GT Windows | Top-1 Time | Score | Hit@1 | Hit@5 | MRR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 9769 | `j7rJstUseKg_360.0_510.0` | some military patriots takes us through ... | [[72, 82], [84, 94], [96, 106], [108, 118], [120, 130], [136, 142], [144, 146]] | 02:02 | 0.1074 | ✅ | ✅ | 1.0 |
| 10016 | `j7rJstUseKg_210.0_360.0` | Man in baseball cap eats before doing hi... | [[96, 114]] | 02:09 | 0.0791 | ❌ | ✅ | 0.3333 |
| 10078 | `j7rJstUseKg_60.0_210.0` | A man in a white shirt discusses the rig... | [[48, 50], [76, 120], [122, 138], [140, 146]] | 02:04 | 0.0852 | ✅ | ✅ | 1.0 |
| 6812 | `-Oc6gSWB_HA_60.0_210.0` | A view of a bamboo fountain of water in ... | [[64, 92]] | 01:26 | 0.0724 | ✅ | ✅ | 1.0 |
| 9247 | `G60-kHBEeZA_60.0_210.0` | The weather map shows large snowfall in ... | [[14, 48]] | 00:33 | 0.1267 | ✅ | ✅ | 1.0 |
| 3563 | `Ok-M_V_h-eY_360.0_510.0` | The view from the car window as it drive... | [[14, 40]] | 00:27 | 0.0891 | ✅ | ✅ | 1.0 |
| 6804 | `Ok-M_V_h-eY_60.0_210.0` | A woman is approaching an illuminate ent... | [[120, 134]] | 02:09 | 0.0823 | ✅ | ✅ | 1.0 |
| 126 | `C0MQLh8Az7U_60.0_210.0` | Woman sits in the drivers seat of her ca... | [[6, 40]] | 00:06 | 0.1125 | ✅ | ✅ | 1.0 |
| 468 | `C0MQLh8Az7U_360.0_510.0` | Kids pours liquid from a purple box into... | [[30, 56]] | 00:36 | 0.1677 | ✅ | ✅ | 1.0 |
| 6170 | `C0MQLh8Az7U_210.0_360.0` | A young woman prepares some avocado toas... | [[84, 120]] | 01:54 | 0.1436 | ✅ | ✅ | 1.0 |
| 87 | `S-nHYzK-BVg_360.0_510.0` | A skeleton ociture is in a word document... | [[130, 142], [148, 150]] | 02:27 | 0.126 | ❌ | ✅ | 0.5 |
| 1250 | `S-nHYzK-BVg_60.0_210.0` | Different menus are shown on a computer ... | [[2, 16], [106, 112], [136, 140]] | 01:46 | -0.0056 | ✅ | ✅ | 1.0 |
| 3986 | `S-nHYzK-BVg_210.0_360.0` | Computer screen is showing a word doc.... | [[0, 150]] | 00:00 | -0.0126 | ✅ | ✅ | 1.0 |
| 7165 | `S-nHYzK-BVg_510.0_660.0` | Picture of skeleton is on a word documen... | [[0, 76], [80, 150]] | 01:24 | 0.1213 | ✅ | ✅ | 1.0 |
| 8265 | `S-nHYzK-BVg_660.0_810.0` | A demonstration of how to use photo soft... | [[0, 24]] | 01:19 | 0.156 | ❌ | ❌ | 0.1667 |
| 4569 | `W8V0z-_dadg_360.0_510.0` | Man walks everywhere in a black t shirt.... | [[0, 24], [28, 32], [40, 48], [52, 60], [70, 84], [104, 110], [120, 126], [130, 134], [136, 144]] | 00:55 | 0.1136 | ✅ | ✅ | 1.0 |
| 7908 | `W8V0z-_dadg_210.0_360.0` | Man sets up his workspace then sits and ... | [[100, 116]] | 01:53 | 0.0833 | ✅ | ✅ | 1.0 |
| 8118 | `W8V0z-_dadg_60.0_210.0` | People are working out at the gym.... | [[96, 122]] | 01:55 | 0.1162 | ✅ | ✅ | 1.0 |
| 5065 | `LoLqz33MJNU_210.0_360.0` | Two asian chefs cooking a meal... | [[44, 56], [60, 78], [88, 136], [146, 148]] | 00:17 | 0.1237 | ❌ | ✅ | 0.5 |
| 8328 | `LoLqz33MJNU_510.0_660.0` | Man in black top walks to the table then... | [[136, 150]] | 01:13 | 0.0557 | ❌ | ❌ | 0.1 |
| 8860 | `LoLqz33MJNU_660.0_810.0` | Two men share a bowl of noodles... | [[0, 118]] | 01:21 | 0.1041 | ✅ | ✅ | 1.0 |
| 9782 | `bJZ-FTeG7D8_60.0_210.0` | Man is wearing a headset while interview... | [[0, 150]] | 00:00 | 0.0869 | ✅ | ✅ | 1.0 |
| 9785 | `bJZ-FTeG7D8_210.0_360.0` | Two people from the same show interview ... | [[0, 150]] | 00:00 | 0.069 | ✅ | ✅ | 1.0 |
| 10219 | `bJZ-FTeG7D8_360.0_510.0` | Man is wearing a headset during a interv... | [[0, 150]] | 00:00 | 0.0851 | ✅ | ✅ | 1.0 |
| 9644 | `WmId2ZP3h0c_360.0_510.0` | Man and woman having interview about ind... | [[4, 20], [28, 30], [32, 46]] | 00:29 | 0.0852 | ✅ | ✅ | 1.0 |
| 9739 | `WmId2ZP3h0c_60.0_210.0` | A group of girls and boys are dancing.... | [[18, 38]] | 01:05 | 0.0782 | ❌ | ✅ | 0.5 |
| 10231 | `WmId2ZP3h0c_210.0_360.0` | Students wearing gender colored red and ... | [[98, 114]] | 01:49 | 0.1433 | ✅ | ✅ | 1.0 |
| 1287 | `hu5L0-CnuUw_360.0_510.0` | A spoon stirs the yellow curry in a pot.... | [[36, 68]] | 00:27 | 0.11 | ❌ | ❌ | 0.1429 |
| 2588 | `hu5L0-CnuUw_210.0_360.0` | A little girl showing her hair and talki... | [[122, 150]] | 02:27 | 0.1112 | ✅ | ✅ | 1.0 |
| 5122 | `hu5L0-CnuUw_60.0_210.0` | Mom does her daughter's makeup.... | [[50, 56], [66, 76], [102, 112]] | 00:39 | 0.1115 | ❌ | ✅ | 0.5 |
| 6931 | `hu5L0-CnuUw_660.0_810.0` | Little girl rides in the car with her mo... | [[52, 76]] | 00:54 | 0.0706 | ✅ | ✅ | 1.0 |
| 8766 | `hu5L0-CnuUw_510.0_660.0` | A woman in green blouse is talking in fr... | [[38, 110]] | 00:40 | 0.0841 | ✅ | ✅ | 1.0 |
| 7824 | `q8YvpLSXQnk_210.0_360.0` | Two women in bathing suits walk along th... | [[0, 18], [30, 34]] | 01:21 | 0.0829 | ❌ | ✅ | 0.3333 |
| 5607 | `RZrf7QFxIW4_60.0_210.0` | Woman writes in a notebook.... | [[94, 108]] | 01:36 | 0.1161 | ✅ | ✅ | 1.0 |
| 5803 | `RZrf7QFxIW4_210.0_360.0` | Masked woman walks around with her hood ... | [[126, 136], [144, 150]] | 02:08 | 0.1298 | ✅ | ✅ | 1.0 |
| 7771 | `RZrf7QFxIW4_360.0_510.0` | A computer screen with text is shown.... | [[12, 26]] | 00:20 | 0.0967 | ✅ | ✅ | 1.0 |
| 285 | `9kwlibPTcJU_210.0_360.0` | Food is in tupperware and on plates.... | [[66, 82]] | 01:15 | 0.0964 | ✅ | ✅ | 1.0 |
| 350 | `9kwlibPTcJU_360.0_510.0` | A girl is having fun with her friends... | [[0, 72]] | 00:27 | 0.1074 | ✅ | ✅ | 1.0 |
| 3821 | `9kwlibPTcJU_60.0_210.0` | Blonde woman brushes her face.... | [[46, 50], [52, 56], [58, 60], [64, 68], [70, 72], [76, 80], [84, 86], [90, 92], [96, 102]] | 01:09 | 0.1178 | ❌ | ✅ | 0.5 |
| 7298 | `9kwlibPTcJU_510.0_660.0` | Two female classmates are joking around ... | [[0, 28]] | 01:48 | 0.0488 | ❌ | ✅ | 0.5 |
| 8836 | `9kwlibPTcJU_660.0_810.0` | The lady with white shirt on is trying t... | [[2, 24]] | 00:21 | 0.0938 | ✅ | ✅ | 1.0 |
| 9971 | `Mb_8DJF6Hp0_60.0_210.0` | Two men in hard hats have a conversation... | [[2, 22], [24, 120], [126, 144]] | 00:28 | 0.1241 | ✅ | ✅ | 1.0 |
| 80 | `lAigvAeOoqQ_60.0_210.0` | Gil trying out different outfits at shop... | [[0, 150]] | 00:00 | 0.1084 | ✅ | ✅ | 1.0 |
| 3911 | `lAigvAeOoqQ_360.0_510.0` | A girl talking while getting her make up... | [[0, 144]] | 00:23 | 0.1108 | ✅ | ✅ | 1.0 |
| 6012 | `lAigvAeOoqQ_210.0_360.0` | Camilla Cabello introduces her hair and ... | [[78, 130]] | 02:16 | 0.0836 | ❌ | ✅ | 0.5 |
| 9599 | `Kn1iFDv9Viw_60.0_210.0` | A child lays on their back as a woman pl... | [[52, 54], [120, 130], [132, 140], [144, 148]] | 02:16 | 0.1074 | ✅ | ✅ | 1.0 |
| 568 | `JcHK1SmwDds_360.0_510.0` | Woman is trying to cool off a man in a y... | [[52, 58], [62, 78]] | 02:23 | 0.1251 | ❌ | ✅ | 0.25 |
| 1553 | `JcHK1SmwDds_60.0_210.0` | Youtuber talking about his new tv commer... | [[0, 64]] | 00:06 | 0.0686 | ✅ | ✅ | 1.0 |
| 2811 | `JcHK1SmwDds_210.0_360.0` | Man in yellow top films his medical appo... | [[56, 150]] | 02:04 | 0.0849 | ✅ | ✅ | 1.0 |
| 7516 | `JcHK1SmwDds_510.0_660.0` | A man is going through and tearing away ... | [[70, 92]] | 01:10 | 0.0464 | ✅ | ✅ | 1.0 |
| 155 | `QndZGvTthvY_210.0_360.0` | Dog is on a leash while interacting with... | [[70, 84], [126, 146]] | 01:05 | 0.1139 | ❌ | ✅ | 0.5 |
| 4086 | `QndZGvTthvY_60.0_210.0` | A woman preparing an avocado sandwich... | [[74, 116]] | 01:47 | 0.0874 | ✅ | ✅ | 1.0 |
| 5307 | `QndZGvTthvY_360.0_510.0` | Woman with towel on her head has on a re... | [[62, 76]] | 01:41 | 0.0939 | ❌ | ✅ | 0.25 |
| 7306 | `QndZGvTthvY_660.0_810.0` | Woman picks up a jar of JIF.... | [[8, 22]] | 00:19 | 0.0796 | ✅ | ✅ | 1.0 |
| 8753 | `QndZGvTthvY_510.0_660.0` | A woman with a scarf is walking outside ... | [[122, 150]] | 02:21 | 0.0887 | ✅ | ✅ | 1.0 |
| 1772 | `OKYptIe8a-k_360.0_510.0` | A close up of the pot with the food bein... | [[56, 82], [96, 108]] | 00:32 | 0.0853 | ❌ | ✅ | 0.25 |
| 3878 | `OKYptIe8a-k_60.0_210.0` | A child is cleaning up and organizing hi... | [[14, 40]] | 01:19 | 0.0944 | ❌ | ❌ | 0.125 |
| 5285 | `OKYptIe8a-k_210.0_360.0` | A woman peels and chops a banana for her... | [[24, 56], [118, 130]] | 00:03 | 0.0553 | ❌ | ✅ | 0.5 |
| 6997 | `OKYptIe8a-k_660.0_810.0` | Woman takes off her glasses and puts the... | [[50, 64]] | 00:23 | 0.1005 | ❌ | ✅ | 0.5 |
| 8384 | `OKYptIe8a-k_510.0_660.0` | A mom and jer son eat as they watch TV t... | [[0, 34]] | 00:00 | 0.0767 | ✅ | ✅ | 1.0 |
| 240 | `xrT84MJxBhs_210.0_360.0` | Phone screen showing an ad.... | [[86, 110]] | 00:03 | 0.0316 | ❌ | ❌ | 0.0 |
| 1272 | `xrT84MJxBhs_360.0_510.0` | The girl is sitting cross legged on a ch... | [[104, 126]] | 01:56 | 0.1204 | ✅ | ✅ | 1.0 |
| 1682 | `xrT84MJxBhs_60.0_210.0` | A girl eating her breakfast with tea... | [[0, 26]] | 02:15 | 0.0901 | ❌ | ✅ | 0.5 |
| 8297 | `xrT84MJxBhs_510.0_660.0` | A woman vlogger with white top is showin... | [[64, 80]] | 02:07 | 0.0363 | ❌ | ✅ | 0.5 |
| 8613 | `xrT84MJxBhs_660.0_810.0` | A woman in white t-shirt is talking abou... | [[0, 124]] | 00:25 | 0.0485 | ✅ | ✅ | 1.0 |
| 9876 | `O16JP0YRKv4_210.0_360.0` | Woman interviews soldiers under a tent.... | [[78, 98]] | 01:29 | 0.0812 | ✅ | ✅ | 1.0 |
| 10108 | `O16JP0YRKv4_360.0_510.0` | Woman talks into her blue phone.... | [[50, 82], [86, 96]] | 01:08 | 0.141 | ✅ | ✅ | 1.0 |
| 10144 | `O16JP0YRKv4_60.0_210.0` | Soldier is prone looking through his rif... | [[34, 38], [44, 78]] | 00:51 | 0.1469 | ✅ | ✅ | 1.0 |
| 9666 | `0Cen89PVfhE_360.0_510.0` | A man is being interviewed in front of a... | [[54, 74], [102, 150]] | 02:29 | 0.1218 | ✅ | ✅ | 1.0 |
| 9697 | `0Cen89PVfhE_60.0_210.0` | President Trump is giving a speech at a ... | [[92, 112]] | 02:27 | 0.1034 | ❌ | ❌ | 0.1111 |
| 3459 | `sJ-KomL7DUo_60.0_210.0` | A girl enjoying fun water rides... | [[96, 102], [106, 150]] | 02:12 | 0.1245 | ✅ | ✅ | 1.0 |
| 7910 | `sJ-KomL7DUo_360.0_510.0` | A stack of luggage is on the floor.... | [[88, 102]] | 01:35 | 0.0669 | ✅ | ✅ | 1.0 |
| 8105 | `sJ-KomL7DUo_210.0_360.0` | Woman in striped shift is riding a inner... | [[2, 4], [8, 12], [20, 26], [28, 38], [40, 44], [46, 50], [54, 60]] | 00:21 | 0.1014 | ✅ | ✅ | 1.0 |
| 34 | `TUq8vM0pRO8_60.0_210.0` | A man in a blue sleeveless shirt  is sho... | [[134, 150]] | 02:14 | 0.0515 | ✅ | ✅ | 1.0 |
| 1871 | `ZQ50DVAjzyQ_360.0_510.0` | a man with british accent is on a boat w... | [[40, 46], [52, 58], [62, 64], [66, 68], [78, 80], [86, 92], [94, 96], [104, 106]] | 00:45 | 0.0005 | ✅ | ✅ | 1.0 |
| 4321 | `ZQ50DVAjzyQ_210.0_360.0` | Woman and man both wear hats during thei... | [[46, 78], [80, 82], [86, 88]] | 00:59 | 0.1139 | ✅ | ✅ | 1.0 |
| 8015 | `ZQ50DVAjzyQ_60.0_210.0` | A couple is sharing a tall yellow ice cr... | [[84, 104]] | 01:36 | 0.119 | ✅ | ✅ | 1.0 |
| 71 | `l6nk_fbGXeo_60.0_210.0` | Mother holds up a bag of doll clothes.... | [[130, 132], [134, 142], [144, 150]] | 02:18 | 0.0891 | ✅ | ✅ | 1.0 |
| 176 | `l6nk_fbGXeo_360.0_510.0` | Kids are playing on the stairs.... | [[2, 36]] | 00:01 | 0.1451 | ❌ | ✅ | 0.5 |
| 3435 | `S1Xq6MbAao0_60.0_210.0` | Some close ups of a starter at a restaur... | [[78, 122]] | 01:37 | 0.0455 | ✅ | ✅ | 1.0 |
| 7632 | `S1Xq6MbAao0_210.0_360.0` | The end of the video telling viewers to ... | [[118, 136]] | 01:58 | 0.0431 | ✅ | ✅ | 1.0 |
| 1418 | `7YdNHlbtMI8_210.0_360.0` | A guy unpacking his baseball hand guards... | [[10, 68], [72, 74]] | 00:35 | 0.1217 | ✅ | ✅ | 1.0 |
| 1588 | `7YdNHlbtMI8_60.0_210.0` | A man practicing baseball throwing... | [[86, 92], [112, 128]] | 01:58 | 0.0935 | ✅ | ✅ | 1.0 |
| 2410 | `7YdNHlbtMI8_360.0_510.0` | Base Ball player showing some practice s... | [[34, 150]] | 01:35 | 0.1075 | ✅ | ✅ | 1.0 |
| 8592 | `7YdNHlbtMI8_510.0_660.0` | Baseball player participates in a profes... | [[46, 88]] | 01:24 | 0.0868 | ✅ | ✅ | 1.0 |
| 323 | `i3mGaC0d-lA_210.0_360.0` | Old woman is sitting on a striped blanke... | [[6, 22]] | 00:51 | 0.0207 | ❌ | ✅ | 0.25 |
| 1227 | `i3mGaC0d-lA_360.0_510.0` | Indian boys with similar check shirts ha... | [[0, 34]] | 00:29 | 0.1027 | ✅ | ✅ | 1.0 |
| 2416 | `i3mGaC0d-lA_60.0_210.0` | A child is playing with dogs and massagi... | [[16, 48], [52, 60]] | 00:25 | 0.0807 | ✅ | ✅ | 1.0 |
| 7554 | `i3mGaC0d-lA_510.0_660.0` | People are showing a square shaped plant... | [[98, 112]] | 01:46 | 0.0991 | ✅ | ✅ | 1.0 |
| 3371 | `4ZxgBQA7cuY_210.0_360.0` | The mountainous region around Tiahuanaco... | [[0, 26]] | 00:08 | 0.0783 | ✅ | ✅ | 1.0 |
| 4760 | `4ZxgBQA7cuY_60.0_210.0` | Footage of the inside and outside of an ... | [[68, 98]] | 01:18 | 0.0602 | ✅ | ✅ | 1.0 |
| 1068 | `EnOr9DiGkzU_60.0_210.0` | Vlogger leaves the plane and enters a ta... | [[6, 8], [14, 30]] | 00:14 | 0.1256 | ✅ | ✅ | 1.0 |
| 2195 | `EnOr9DiGkzU_210.0_360.0` | tourist dancing and have photo session o... | [[108, 144]] | 01:53 | 0.1148 | ✅ | ✅ | 1.0 |
| 9096 | `EnOr9DiGkzU_510.0_660.0` | Woman in life vest rolls on the ground t... | [[116, 132]] | 01:59 | 0.1286 | ✅ | ✅ | 1.0 |
| 4862 | `4vBHEeKjWJ8_360.0_510.0` | Far away view of a man sitting on grassy... | [[0, 22]] | 00:31 | 0.0923 | ❌ | ✅ | 0.25 |
| 5100 | `4vBHEeKjWJ8_60.0_210.0` | Man cooks himself some dinner.... | [[110, 126], [144, 150]] | 01:52 | 0.1126 | ✅ | ✅ | 1.0 |
| 9168 | `jabvgI8GZjM_60.0_210.0` | Hurricane Iota has taken over all the pl... | [[0, 24], [92, 102], [104, 108], [120, 122], [126, 128], [130, 144]] | 01:59 | 0.0984 | ❌ | ✅ | 0.5 |
| 1905 | `Mw1mWje-558_210.0_360.0` | A man explains how to download songs for... | [[86, 96], [98, 108], [110, 112]] | 02:07 | 0.0954 | ❌ | ❌ | 0.1429 |
| 2143 | `FCjdUpAzqjA_210.0_360.0` | Two young women walk down a Tokyo street... | [[42, 58], [68, 84]] | 00:53 | 0.0682 | ✅ | ✅ | 1.0 |
| 7963 | `FCjdUpAzqjA_60.0_210.0` | Woman is walking through  convenience st... | [[94, 108]] | 01:35 | 0.1038 | ✅ | ✅ | 1.0 |

---
*Full per-query raw logs and metrics are exported in `evaluation_details.csv`.*  
*Retrieved frames can be inspected in `retrieved_frames/`.*
