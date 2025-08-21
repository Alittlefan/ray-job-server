import json


def append_dict_to_json(data, filename):
    with open(filename, "a", encoding="utf-8") as f:
        json_str1 = json.dumps(dict(data))
        f.write(json_str1 + "\n")
        f.close()


def json_load(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def parse_tuple(string):
    return tuple(map(float, string.strip("()").split(",")))


def tongshi_get_pos(data, category, n):
    """
    n : 获取数量
    """
    red_positions = []
    blue_positions = []
    items = data.get(category, [])
    if 0 <= n < len(items):
        nth_item = items[n]
        red_positions.append(parse_tuple(nth_item["red_pos"]))
        blue_positions.extend([parse_tuple(pos) for pos in nth_item["blue_pos"]])
    else:
        print(f"指定索引{n}超出索引范围")
    return red_positions, blue_positions


def append_frame_to_txt(data, filename):
    json_str = json.dumps(data, ensure_ascii=False)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json_str + "\n")


def load_json_line(file_name):
    init_pos_result = []
    dead_pos_result = []
    with open(file_name, "r") as f:
        for line in f:
            epoch_data = json.loads(line)
            if len(epoch_data["blue_dead_pos"]) != 0:
                for i in range(len(epoch_data["blue_dead_pos"])):
                    init_pos_result.append(
                        [
                            epoch_data["dead_from"][i][1][1],
                            epoch_data["dead_from"][i][1][0],
                            epoch_data["dead_from"][i][1][2],
                            0,
                        ]
                    )
                    dead_pos_result.append(
                        [
                            epoch_data["blue_dead_pos"][i][2],
                            epoch_data["blue_dead_pos"][i][1],
                            epoch_data["blue_dead_pos"][i][3],
                            0,
                        ]
                    )

    return init_pos_result, dead_pos_result


def write_json(file_path, dict):
    with open(file_path, "w") as f:
        json_str1 = json.dumps(dict)
        f.write(json_str1 + "\n")
        f.close()


def save_dict_to_json(dict, file_path, indent=4, ensure_ascii=False):
    """
    规范的写入json
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(dict, file, indent=indent, ensure_ascii=ensure_ascii)
        print("成功保存")
        return True


###ceshi

# def main():
#     file_name = "data_collect\\normal_msg\\duo_bigua_msg.json"
#     init_pos_result ,dead_pos_sult = load_json_line(file_name)


# if __name__ == '__main__':
#     main()
