import json
import pandas as pd    


def main():
    input_file = "data/na_dirty.jsonl"
    output_file = "data/na_eval.json"

    
    # query_data = pd.read_json(path_or_buf=input_file, lines=True)
    # print(query_data.info())
    
    with open(input_file, 'r') as json_file:
        json_list = list(json_file)

    query_dicts = []
    for json_str in json_list:
        result = json.loads(json_str)
        query_dicts.append(result)


    clean_data = []
    for query_dict in query_dicts:
        clean_dict = {}
        clean_dict['id'] = query_dict['id']
        clean_dict['question'] = query_dict['input']

        output = query_dict['output']
        answers = []
        kilt_provenance = []
        # print(id)
        for l in output:
            answer = l['answer']
            provenance = l['provenance']
            if answer:
                answers.append(answer)

            if provenance:
                clean_provenance = provenance[0]
                clean_provenance['answer'] = answer
                clean_provenance['has_answer'] = bool(answer)
                kilt_provenance.append(clean_provenance)

        clean_dict['answers'] = answers
        clean_dict['kilt_provenance'] = kilt_provenance
        clean_data.append(clean_dict)

    with open(output_file, "w") as f:
        json.dump(clean_data, f, indent=4)

if __name__ == "__main__":
    main()