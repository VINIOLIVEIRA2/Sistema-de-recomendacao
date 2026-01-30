import json
from api.app import recommend
print(json.dumps(recommend(123), ensure_ascii=False))
