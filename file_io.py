import csv
import json
import xlrd


def read_txt_file(file_path):
    """Read a txt file then return its lines in a list."""
    with open(file_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    return lines


def read_csv_file(file_path):
    """Read a csv file then return its items in a list."""
    items = list()
    with open(file_path, 'r', encoding='utf-8') as fp:
        csv_reader = csv.reader(fp)
        for item in csv_reader:
            items.append(item)
        return items


def read_tsv_file(file_path):
    """Read a tsv file then return its items in a list."""
    csv.register_dialect('tsv', delimiter='\t')
    items = list()
    with open(file_path, 'r', encoding='utf-8') as fp:
        tsv_reader = csv.reader(fp, 'tsv')
        for item in tsv_reader:
            items.append(item)
        return items


def read_json_file(file_path):
    """Read a json file then return its dicts in a list."""
    with open(file_path, 'r', encoding='utf-8') as fp:
        dicts = json.load(fp)
        return dicts


def read_jsonl_file(file_path):
    """Read a jsonl file then return its dicts in a list."""
    with open(file_path, "r", encoding="utf-8") as fp:
        dicts = [json.loads(line.strip()) for line in fp]
        return dicts


def read_xlsx_file(file_path, sheet_index=0):
    """Read a xlsx file then return its table in a list."""
    table = list()
    work_book = xlrd.open_workbook(file_path)
    work_sheet = work_book.sheet_by_index(sheet_index)
    for row in range(work_sheet.nrows):
        table.append(work_sheet.row_values(row))
    return table


def write_txt_file(file_path, lines):
    """Write a list of lines to a txt file."""
    with open(file_path, 'w', encoding='utf-8', newline='') as fp:
        for line in lines:
            fp.write(line + '\n')


def write_csv_file(file_path, items):
    """Write a list of items to a csv file."""
    with open(file_path, 'w', encoding='utf-8', newline='') as fp:
        csv_writer = csv.writer(fp)
        for item in items:
            csv_writer.writerow(item)


def write_tsv_file(file_path, items):
    """Write a list of items to a tsv file."""
    csv.register_dialect('tsv', delimiter='\t')
    with open(file_path, 'w', encoding='utf-8', newline='') as fp:
        tsv_writer = csv.writer(fp, 'tsv')
        for item in items:
            tsv_writer.writerow(item)


def write_json_file(file_path, dicts):
    """Write a list of dicts to a json file."""
    with open(file_path, 'w', encoding='utf-8', newline='') as fp:
        json.dump(dicts, fp, indent=2, ensure_ascii=False)
