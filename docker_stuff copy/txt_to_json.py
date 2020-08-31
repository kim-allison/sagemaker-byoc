import json
with open("products.txt", 'r', encoding='utf-8-sig') as f:
    lst_text = f.readlines()
    lst_product = []

    for i in range(0,len(lst_text)):
        if i%4 == 1:
            product = lst_text[i][3:]
            lst_product.append(product)

    products = {}
    products["products"] = lst_product
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(products, f, ensure_ascii=False, indent=4)