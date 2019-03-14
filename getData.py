import pymysql

cnRegion = ['AH', 'BJ', 'CQ', 'FJ', 'GD', 'GS', 'GX', 'GZ', 'HA', 'HB', 'HE', 'HI', 'HK', 'HL', 'HN', 'JL', 'JS', 'JX', 'LN', 'MO', 'NM', 'NX', 'QH',
'SC', 'SD', 'SH', 'SN', 'SX', 'TJ', 'TW', 'XJ', 'XZ', 'YN', 'ZJ']

overseaRegion = ['Aceh', 'Afghanistan', 'Africa', 'Aisa', 'Alabama', 'Alaska', 'Albania', 'Alberta', 'Algeria', 'AndhraPradesh', 'Angola',
 'Argentina', 'Arizona', 'Arkansas', 'Armenia', 'ArunachalPradesh', 'Assam', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bangkok', 
 'Bangladesh', 'Banten', 'Belarus', 'Belgium', 'Belize', 'Bengkulu', 'Benin', 'Bhutan', 'Bihar', 'Bolivia', 'Bosnia And Herzegovina', 
 'Botswana', 'Brazil', 'British Columbia', 'Brunei Darussalam', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'CenteralJava', 
 'Central African Republic', 'Chad', 'Chandigarh', 'Chhattisgarh', 'Chile', 'China', 'Colombia', 'Colorado', 'Congo', 'Connecticut', 'Costa Rica', 
 "Cote D'Ivoire", 'Croatia/Hrvatska', 'Cuba', 'Cyprus', 'Czech Republic', 'DadraAndNagarHaveli', 'Delaware', 'Denmark', 'Djibouti', 
 'Dominican Republic', 'Dubai', 'EastJava', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia',
 'Europe', 'Falkland Islands (Malvinas)', 'Fiji', 'Finland', 'Florida', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 
 'Goa', 'Greece', 'Greenland', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Gujarat', 'Guyana', 'Haiti', 'Haryana', 'Hawaii', 'HimachalPradesh',
 'Ho Chi Minh City', 'Honduras', 'Hungary', 'Iceland', 'Idaho', 'Illinois', 'India', 'Indiana', 'Indonesia', 'Iowa', 'Iran', 'Iraq', 'Ireland',
 'Israel', 'Italy', 'Jakarta', 'Jamaica', 'Jambi', 'JammuAndKashmir', 'Japan', 'Jharkhand', 'Jordan', 'Kansas', 'Karnataka', 'Kazakhstan',
 'Kentucky', 'Kenya', 'Kerala', 'Korea', 'Kuwait', 'Kyrgyzstan', 'Lampgug', "Lao People's Democratic Republic", 'Latvia', 'Lebanon', 
 'Leningrad Region', 'Lesotho', 'Liberia', 'Libyan Arab Jamahiriya', 'Lithuania', 'Local Area Network', 'London', 'Louisiana', 
 'Luxembourg', 'Macedonia', 'Madagascar', 'MadhyaPradesh', 'Madrid', 'Maharashtra', 'Maine', 'Malawi', 'Malaysia', 'Mali', 
 'Manipur', 'Manitoba', 'Maryland', 'Massachusetts', 'Mauritania', 'Meghalaya', 'Mexico', 'Michigan', 'Milan', 'Minnesota', 'Mississippi', 
 'Missouri', 'Mizoram', 'Moldova', 'Mongolia', 'Montana', 'Montenegro', 'Morocco', 'Moscow', 'Mozambique', 'Myanmar', 'Nagaland',
 'NamanAndDiu', 'Namibia', 'Nebraska', 'Negeri Selangor', 'Nepal', 'Netherlands', 'Nevada', 'New Caledonia', 'New Zealand', 
 'New-Brunswick', 'NewDelhi', 'Newfoundland-and-Labrador', 'NewHampshire', 'NewJersey', 'NewMexico', 'NewYork', 'Nicaragua', 'Niger', 
 'Nigeria', 'North America', 'North-California', 'NorthCarolina', 'NorthDakota', 'NorthSumatra', 'Northwest-Territories', 'Norway', 
 'Nova-Scotia', 'Nunavut', 'Oceania', 'Odisha', 'Ohio', 'Oklahoma', 'Oman', 'Ontario', 'Oregon', 'Osaka', 'Oversea', 'Pakistan', 
 'Palestinian Territory', 'Panama', 'Papua New Guinea', 'Paraguay', 'Pennsylvania', 'Peru', 'Philippines', 'Poland', 'Portugal',
 'Prince-Edward-Island', 'Puducherry', 'Puerto Rico', 'Punjab', 'Qatar', 'Quebec', 'Rajasthan', 'RhodeIsland', 'Riau', 'Romania', 
 'Russian Federation', 'Rwanda', 'Sao Paulo', 'Saskatchewan', 'Saudi Arabia', 'Senegal', 'Seoul Special', 'Serbia', 'Sierra Leone',
 'Sikkim', 'Singapore', 'Slovakia (Slovak Republic)', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South America', 
 'South-California', 'SouthCarolina', 'SouthDakota', 'SouthSumatra', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden',
 'Switzerland', 'Syrian Arab Republic', 'Tajikistan', 'TamilNadu', 'Tanzania', 'Telangana', 'Tennessee', 'Texas', 'Thailand', 
 'Timor-Leste', 'Togo', 'Tokyo', 'Tripura', 'Tunisia', 'Turkey', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates',
 'United Kingdom', 'United States', 'Uruguay', 'Utah', 'Uttarakhand', 'UttarPradesh', 'Uzbekistan', 'Vanuatu', 'Venezuela', 
 'Vermont', 'Vietnam', 'Virginia', 'Washington', 'WashingtonDC', 'WestBengal', 'WestJava', 'WestSumatra', 'WestVirginia',
 'Wisconsin', 'Wyoming', 'Yemen', 'Yogyakarta', 'Yukon', 'Zambia', 'Zimbabwe']

cnHost = ['.alpha-browser.com', '.amemv.com', '.biubiuapp.net', '.bytedance.com', '.bytedance.fj.cn', '.bytedance.net', '.dcdapp.com', 
'.douyin.com', '.douyinact.com', '.douyinact.net', '.douyincdn.com', '.huoshan.com', '.huoshanvideo.cn', '.huoshanvideo.net', 
'.huoshanzhibo.cn', '.huoshanzhibo.com', '.huoshanzhibo.net', '.iesdouyin.net', '.ieshuodong.cn', '.ieshuodong.net', '.ixigua.com', 
'.jinritemai.com', '.jinritoutiao.bj.cn', '.jinritoutiao.js.cn', '.jisukandian.com', '.kachacamera.com', '.luckycalendar.cn',
'.neihanshequ.com', '.quanquanapp.net', '.quduzixun.com', '.ribaoapi.com', '.snssdk.com', '.stock.tuchong.com', '.toutiao.com', 
'.tuchong.com', '.wenxingonline.com', '.wukongwenda.cn', 'abn.snssdk.com', 'ad.toutiao.com', 'admin.stock.tuchong.com',
'api-intl1.huoshan.com', 'api.huoshan.com', 'api.zjurl.cn', 'app.toutiaoribao.cn', 'auto.365yg.com', 'aweme.snssdk.com',
'blog.ixigua.com', 'c.ixigua.com', 'creator.ixigua.com', 'csp.snssdk.com', 'd.amemv.com', 'd.douyin.com', 'd.toutiaoribao.cn', 
'danpin.snssdk.com', 'dcdapp.com', 'dm.ribaoapi.com', 'dm.toutiaoribao.cn', 'extlog.snssdk.com', 'haohuo.jinritemai.com',
'haohuo.snssdk.com', 'i.365yg.com', 'ib.365yg.com', 'ichannel.snssdk.com', 'is.snssdk.com', 'isub.ribaoapi.com', 
'isub.snssdk.com', 'kaidian.jinritemai.com', 'kaidian.snssdk.com', 'lf.ribaoapi.com', 'lf.snssdk.com', 'log.ribaoapi.com', 
'log.snssdk.com', 'm.ixigua.com', 'm.toutiao.com', 'm.toutiao11.com', 'm.toutiao12.com', 'm.toutiao13.com', 'm.toutiao14.com', 
'm.toutiao15.com', 'm.toutiaocdn.cn', 'm.toutiaocdn.com', 'm.toutiaocdn.net', 'm.toutiaoimg.cn', 'm.toutiaoimg.com', 'm.toutiaoimg.net',
'm.toutiaoribao.cn', 'm.xiguaapp.cn', 'm.xiguaapp.com', 'm.xiguashipin.cn', 'm.xiguashipin.net', 'm.xiguavideo.cn', 'm.xiguavideo.net',
'm.zijiecdn.cn', 'm.zijiecdn.com', 'm.zijiecdn.net', 'm.zijieimg.cn', 'm.zijieimg.com', 'm.zijieimg.net', 'm.zjurl.cn', 'm1.toutiao13.com',
'm2.toutiao13.com', 'm3.toutiao13.com', 'mcs.snssdk.com', 'mon.ribaoapi.com', 'mon.snssdk.com', 'monsetting.toutiao.com', 'mp.toutiao.com',
'nativeapp.toutiaoribao.cn', 'open.snssdk.com', 'open.stock.tuchong.com', 'open.toutiao.com', 'reflow.huoshan.com', 's.amemv.com', 
's.douyin.com', 's.toutiaoribao.cn', 'shop.snssdk.com', 'stock.tuchong.com', 't.zijieimg.cn', 't.zijieimg.com', 'temai.snssdk.com',
'test.yun.dfic.cn', 'toblog.snssdk.com', 'wallet.amemv.com', 'wallet.douyin.com', 'www.ixigua.com', 'www.jinritemai.com', 
'www.toutiao.com', 'www.toutiaoribao.cn', 'www.wukong.com', 'xlog.snssdk.com', 'yun.dfic.cn']


overseaHost = ['.alpha-browser.com', '.amemv.com', '.bytedance.com', '.bytedance.fj.cn', '.bytedance.net', '.douyin.com',
 '.douyinact.com', '.douyinact.net', '.douyincdn.com', '.huoshan.com', '.huoshanzhibo.com', '.iesdouyin.net', '.ieshuodong.cn', 
 '.ieshuodong.net', '.ixigua.com', '.jinritemai.com', '.jinritoutiao.bj.cn', '.jinritoutiao.js.cn', '.jisukandian.com', 
 '.kachacamera.com', '.luckycalendar.cn', '.neihanshequ.com', '.quanquanapp.net', '.quduzixun.com', '.ribaoapi.com', 
 '.snssdk.com', '.stock.tuchong.com', '.toutiao.com', '.tuchong.com', '.wenxingonline.com', '.wukongwenda.cn', 'abn.snssdk.com',
 'ad.toutiao.com', 'admin.stock.tuchong.com', 'api.huoshan.com', 'api.zjurl.cn', 'app.toutiaoribao.cn', 'auto.365yg.com', 
 'aweme.snssdk.com', 'blog.ixigua.com', 'c.ixigua.com', 'creator.ixigua.com', 'csp.snssdk.com', 'd.amemv.com', 'd.douyin.com',
 'd.toutiaoribao.cn', 'danpin.snssdk.com', 'dcdapp.com', 'dm.ribaoapi.com', 'dm.toutiaoribao.cn', 'extlog.snssdk.com', 'haohuo.jinritemai.com',
 'haohuo.snssdk.com', 'ib.365yg.com', 'ichannel.snssdk.com', 'is.snssdk.com', 'isub.ribaoapi.com', 'isub.snssdk.com', 'kaidian.jinritemai.com',
 'lf.ribaoapi.com', 'lf.snssdk.com', 'log.ribaoapi.com', 'log.snssdk.com', 'm.ixigua.com', 'm.toutiao.com', 'm.toutiao11.com', 'm.toutiao12.com',
 'm.toutiao13.com', 'm.toutiaocdn.cn', 'm.toutiaocdn.com', 'm.toutiaocdn.net', 'm.toutiaoimg.cn', 'm.toutiaoimg.com', 'm.toutiaoimg.net',
 'm.toutiaoribao.cn', 'm.xiguaapp.cn', 'm.xiguaapp.com', 'm.xiguashipin.cn', 'm.xiguashipin.net', 'm.xiguavideo.cn', 'm.xiguavideo.net',
 'm.zijiecdn.cn', 'm.zijiecdn.com', 'm.zijiecdn.net', 'm.zijieimg.net', 'm.zjurl.cn', 'm2.toutiao13.com', 'mcs.snssdk.com', 'mon.ribaoapi.com',
 'mon.snssdk.com', 'monsetting.toutiao.com', 'mp.toutiao.com', 'open.snssdk.com', 'open.toutiao.com', 'reflow.huoshan.com', 's.douyin.com',
 'shop.snssdk.com', 'stock.tuchong.com', 't.zijieimg.com', 'temai.snssdk.com', 'toblog.snssdk.com', 'www.ixigua.com', 'www.toutiao.com',
 'www.toutiaoribao.cn', 'www.wukong.com', 'xlog.snssdk.com', 'yun.dfic.cn']


def dictfetchall(cursor):#返还字段dict
    desc = cursor.description
    return [
        dict(zip([col[0] for col in desc], row))
        for row in cursor.fetchall()
    ]

connection = pymysql.connect("10.17.35.13", "cdn_monitor_readonly_dw", "M0n1t07Dw", "cdn_test_1",charset='utf8')

cursor = connection.cursor()


for region in cnRegion:
	for host in cnHost:
		order = "SELECT avg(opt_value) FROM cdn_monitor_ali_dynamic_20190228  where s_region ='"+region+"' and host = '"+host+"'"
		cursor.execute(order)
		row = dictfetchall(cursor)
		value = row[0]['avg(opt_value)']
		if value is not None:
			value = int(value)
			if value > 10:
				with open("info.txt","a") as f:
					word = str(region)+" "+str(host)+" "+str(value)+"\n"
					f.write(word)
					print(word)