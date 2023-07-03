model=$1
source_lang=$2
target_lang=$3
output_folder=$4
lang_pairs="de-en,hi-en,km-en,lo-en,ne-en,ta-en,te-en"
lang_list="../data/lang-list.txt"
path_2_data="../data/byte-bin/xx"

mkdir $output_folder
CUDA_VISIBLE_DEVICES=3 fairseq-generate $path_2_data \
  --path $model \
  --task translation_multi_simple_epoch \
  --gen-subset test \
  --source-lang $source_lang \
  --target-lang $target_lang \
  --byte-tokens \
  --batch-size 32 \
  --lang-dict "$lang_list" \
  --skip-invalid-size-inputs-valid-test \
  --lang-pairs "$lang_pairs" > ${output_folder}/${source_lang}_${target_lang}.txt


cat ${output_folder}/${source_lang}_${target_lang}.txt | grep -P "^H" |sort -V |cut -f 3- |cat > ${output_folder}/${source_lang}_${target_lang}.hyp 
cat ${output_folder}/${source_lang}_${target_lang}.txt | grep -P "^T" |sort -V |cut -f 2- |cat > ${output_folder}/${source_lang}_${target_lang}.ref

moses_detokenizer="perl examples/translation/mosesdecoder/scripts/tokenizer/detokenizer.perl"
$moses_detokenizer < ${output_folder}/${source_lang}_${target_lang}.ref > ${output_folder}/${source_lang}_${target_lang}.ref.detok ;
mv ${output_folder}/${source_lang}_${target_lang}.ref.detok ${output_folder}/${source_lang}_${target_lang}.ref
$moses_detokenizer < ${output_folder}/${source_lang}_${target_lang}.hyp > ${output_folder}/${source_lang}_${target_lang}.hyp.detok ;
mv ${output_folder}/${source_lang}_${target_lang}.hyp.detok ${output_folder}/${source_lang}_${target_lang}.hyp

sacrebleu ${output_folder}/${source_lang}_${target_lang}.ref --metrics bleu --tokenize 13a <  ${output_folder}/${source_lang}_${target_lang}.hyp >   ${output_folder}/${source_lang}_${target_lang}.bleu