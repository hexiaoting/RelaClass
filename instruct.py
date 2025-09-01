normal_task = "Generate embeddings for these product descriptions, maximizing the distinguishability between different products."

amazon_531_instruct_task123="I want to perform hierarchical text classification, where the first level of the classification consists of 6 categories: grocery gourmet food, toys games, beauty, health personal care, baby products, and pet supplies. \
\
grocery gourmet food includes meat poultry,beverages,gourmet gifts,sauces dips,breakfast foods,pantry staples,cooking baking supplies,fresh flowers live indoor plants,herbs,breads bakery,candy chocolate,snack food,meat seafood,baby food,dairy eggs,produce. \
The subcategory meat poultry includes jerky,seafood,sausages, exotic meats. \
The subcategory beverages includes juices,tea,coffee,soft drinks,powdered drink mixes,energy drinks,hot cocoa,coconut water,water,cocktail mixers,sports drinks. \
The subcategory gourmet gifts includes snack gifts,dessert gifts,tea gifts,cheese gifts,candy gifts,sauces gifts,seafood gifts,coffee gifts,chocolate gifts,fruit gifts,meat gifts,spices gifts. \
The subcategory sauces dips includes sauces,dips. \
The subcategory breakfast foods includes cereals,breakfast cereal bars,toaster pastries. \
The subcategory pantry staples includes sauces,condiments,dried beans,packaged meals side dishes,pasta noodles,oils,canned jarred food,jams. \
The subcategory cooking baking supplies includes chocolate,baking mixes,syrups,sugars sweeteners,nuts seeds,extracts flavoring,sugar substitutes,dessert toppings,flours meals,baking powder,food coloring,pastry decorations,coatings batters. \
The subcategory fresh flowers live indoor plants includes live indoor plants,fresh cut flowers. \
The subcategory herbs includes spices seasonings. \
The subcategory breads bakery includes cakes,cookies,pizza crusts,fresh baked cookies,tortillas,breakfast bakery,breadcrumbs,breads,packaged breads,stuffing,breadsticks. \
The subcategory candy chocolate includes chocolate assortments,licorice,chocolate bars,gum,gummy candies,gummy candy,chocolate truffles,hard candies,suckers lollipops,fudge,chewing gum,chocolate pretzels,bars,jelly beans,mints,taffy,halva,nut clusters,toffee,chocolate covered fruit,caramels,assortments,marshmallows,chocolate covered nuts. \
The subcategory snack food includes chips crisps,cookies,dried fruit raisins,party mix,jerky dried meats,crackers,pork rinds,trail mix,rice cakes,chips,puffed snacks,pretzels,popcorn,nuts seeds,granola bars,raisins,salsas,fruit leather,granola trail mix bars,pudding,dried fruit. \
The subcategory meat seafood includes wild game fowl,seafood,beef,chicken. \
The subcategory baby food includes baby formula,crackers biscuits,cereal,dinners,fruit. \
The subcategory dairy eggs includes cheese,milk,eggs,milk substitutes,ice cream frozen desserts. \
The subcategory produce includes fresh fruits,fresh vegetables. \
 \
toys games includes games,puzzles,arts crafts,action toy figures,dolls accessories,baby toddler toys,learning education,electronics for kids,stuffed animals plush,tricycles,grown up toys,dress up pretend play,novelty gag toys,building toys,sports outdoor play,hobbies,vehicles remote control. \
The subcategory games includes board games,card games,trading card games,travel games,battling tops,tile games,dvd games,stacking games,floor games,game collections,standard playing card decks,handheld games,dice gaming dice,game accessories,game room games. \
The subcategory puzzles includes jigsaw puzzles,d puzzles,pegged puzzles,puzzle accessories,floor puzzles,brain teasers,puzzle play mats. \
The subcategory arts crafts includes drawing painting supplies,drawing sketching tablets,clay dough,blackboards whiteboards,craft kits,easels,printing stamping,stickers,molding sculpting sticks,aprons smocks. \
The subcategory action toy figures includes figures,playsets,statues,accessories. \
The subcategory dolls accessories includes dolls,playsets,dollhouses,doll accessories,dollhouse accessories. \
The subcategory baby toddler toys includes shape sorters,push pull toys,music sound,crib toys attachments,stacking nesting toys,bath toys,rattles,activity play centers,car seat stroller toys,hammering pounding toys,rocking spring ride ons,balls,stuffed animals toys,mirrors,blocks,indoor climbers play structures,toy gift sets,teaching clocks. \
The subcategory learning education includes habitats,science,basic life skills toys,reading writing,geography,flash cards,mathematics counting,electronics. \
The subcategory electronics for kids includes electronic toys,music players karaoke,electronic pets,systems accessories,cameras camcorders,dance mats,personal video players accessories,plug play video games,walkie talkies. \
The subcategory stuffed animals plush includes animals figures,puppets,plush backpacks purses,plush pillows,teddy bears,plush puppets. \
The subcategory tricycles includes scooters wagons. \
The subcategory grown up toys includes die cast toy vehicles. \
The subcategory dress up pretend play includes pretend play,beauty fashion. \
The subcategory novelty gag toys includes toy balls,magic kits accessories,temporary tattoos,gag toys practical jokes,money banks,wind up toys,nesting dolls,miniatures,finger boards finger bikes,novelty spinning tops,viewfinders,prisms kaleidoscopes,flying toys. \
The subcategory building toys includes building sets,stacking blocks,marble runs. \
The subcategory sports outdoor play includes play tents tunnels,sports,inflatable bouncers,pools water fun,sandboxes accessories,gym sets swings,kites wind spinners,sand water tables,blasters foam play,gardening tools,pogo sticks hoppers,marble games,fitness equipment,ball pits accessories,beanbags foot bags,kickball playground balls,yo yos. \
The subcategory hobbies includes model building kits tools,rockets,radio control,scaled model vehicles,trains accessories,slot cars,coin collecting. \
The subcategory vehicles remote control includes play vehicles,play trains railway sets,vehicle playsets,die cast vehicles. \
 \
 \
beauty includes makeup,skin care,bath body,tools accessories,hair care,fragrance. \
The subcategory makeup includes nails,face,body,eyes,makeup remover,lips,makeup sets. \
The subcategory skin care includes face,body,eyes,hands nails,sun,sets,maternity. \
The subcategory bath body includes cleansers,bath,scrubs body treatments,sets,bathing accessories. \
The subcategory tools accessories includes nail tools,cotton swabs,facial steamers,mirrors,makeup brushes tools,bags cases,hair coloring tools. \
The subcategory hair care includes styling products,shampoos,conditioners,hair color,styling tools,hair loss products,hair scalp treatments,shampoo conditioner sets,shampoo plus conditioner,hair relaxers,hair perms texturizers. \
The subcategory fragrance includes women s,men s,children s. \
 \
health personal care includes personal care,nutrition wellness,household supplies,health care,medical supplies equipment,baby child care,sexual wellness. \
The subcategory personal care includes deodorants antiperspirants,shaving hair removal,feminine care,oral hygiene,lip care products,eye care,foot care,body art,ear care. \
The subcategory nutrition wellness includes nutrition bars drinks,vitamins supplements,weight loss products,sports supplements. \
The subcategory household supplies includes household batteries,paper plastic,cleaning tools,laundry,household cleaning,lighters,dishwashing,air fresheners. \
The subcategory health care includes allergy,pill cases splitters,foot care,incontinence,cough cold,stress reduction,family planning contraceptives,pain relievers,first aid,diabetes,digestion nausea,thermometers,massage relaxation,thermometer accessories,smoking cessation,alternative medicine,women s health,therapeutic skin care,sleep snoring,stimulants. \
The subcategory medical supplies equipment includes daily living aids,tests,health monitors,braces,bathroom aids safety,occupational physical therapy aids,beds accessories,mobility aids equipment. \
The subcategory baby child care includes. \
The subcategory sexual wellness includes safer sex,adult toys games,sexual enhancers,sensual delights,bondage gear accessories,novelties,sex furniture. \
 \
baby products includes gear,gifts,feeding,diapering,safety,nursery,bathing skin care,car seats accessories,strollers,pregnancy maternity,potty training,health baby care. \
The subcategory gear includes baby gyms playmats,backpacks carriers,activity centers entertainers,walkers,shopping cart covers,swings,playards,baby seats. \
The subcategory gifts includes albums,keepsakes,toy banks. \
The subcategory feeding includes bottle feeding,breastfeeding,pillows stools,food,highchairs booster seats,gift sets,solid feeding,pacifiers accessories. \
The subcategory diapering includes diaper changing kits,diaper pails refills,cloth diapers,diaper bags,wipes holders,disposable diapers,changing table pads covers,portable changing pads,diaper stackers caddies,cloth diaper accessories. \
The subcategory safety includesbathroom safety,gates doorways,monitors,edge corner guards,cabinet locks straps,rails rail guards,sleep positioners,kitchen safety,outdoor safety,electrical safety,harnesses leashes. \
The subcategory nursery includes furniture,nursery d cor,bedding. \
The subcategory bathing skin care includes skin care,bathing tubs seats,grooming healthcare kits,gift sets,soaps cleansers,non slip bath mats,washcloths towels,bubble bath. \
The subcategory car seats accessories includes car seats,accessories. \
The subcategory strollers includes accessories,joggers,standard,tandem,travel systems,lightweight,prams. \
The subcategory pregnancy maternity includes maternity pillows. \
The subcategory potty training includes potties seats,step stools,training pants,seat covers. \
The subcategory health baby care includes teethers,sun protection. \
 \
pet supplies includes cats,dogs,fish aquatic pets,small animals,birds, bunny rabbit central. \
The subcategory cats includes litter housebreaking,food,toys,treats,beds furniture,health supplies,collars,educational repellents,feeding watering supplies,grooming,carriers strollers,cages,cat flaps. \
The subcategory dogs includes litter housebreaking,food,toys,treats,beds furniture,health supplies,collars,feeding watering supplies,training behavior aids,grooming,houses,carriers travel products,doors,apparel accessories. \
The subcategory fish aquatic pets includes food,health supplies,aquarium lights,pumps filters,test kits,water treatments,aquarium hoods,aquariums,automatic feeders,cleaners,aquarium d cor,aquarium starter kits,aquarium heaters,aquarium stands,hydrometers. \
The subcategory small animals includes food,toys,treats,collars,feeding watering supplies,houses habitats,exercise wheels. \
The subcategory birds includes food,toys,treats,health supplies,feeding watering supplies,cages accessories. \
The subcategory bunny rabbit central includes food,treats,collars,feeding watering supplies,houses habitats,rabbit hutches,carriers,odor stain removers. \
 \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding." 


amazon_531_instruct_task12="I want to perform hierarchical text classification, where the first level of the classification consists of 6 categories: grocery gourmet food, toys games, beauty, health personal care, baby products, and pet supplies. \
grocery gourmet food includes meat poultry,beverages,gourmet gifts,sauces dips,breakfast foods,pantry staples,cooking baking supplies,fresh flowers live indoor plants,herbs,breads bakery,candy chocolate,snack food,meat seafood,baby food,dairy eggs,produce. \
toys games includes games,puzzles,arts crafts,action toy figures,dolls accessories,baby toddler toys,learning education,electronics for kids,stuffed animals plush,tricycles,grown up toys,dress up pretend play,novelty gag toys,building toys,sports outdoor play,hobbies,vehicles remote control. \
beauty includes makeup,skin care,bath body,tools accessories,hair care,fragrance. \
health personal care includes personal care,nutrition wellness,household supplies,health care,medical supplies equipment,baby child care,sexual wellness. \
baby products includes gear,gifts,feeding,diapering,safety,nursery,bathing skin care,car seats accessories,strollers,pregnancy maternity,potty training,health baby care. \
pet supplies includes cats,dogs,fish aquatic pets,small animals,birds, bunny rabbit central. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding." 

amazon_531_instruct_task1="I want to perform hierarchical text classification, where the first level of the classification consists of 6 categories: grocery gourmet food, toys games, beauty, health personal care, baby products, and pet supplies. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding." 


amazon_531_food_instruct_task="I want to perform hierarchical text classification, where the first level of the classification consists of 16 categories:  meat poultry,beverages,gourmet gifts,sauces dips,breakfast foods,pantry staples,cooking baking supplies,fresh flowers live indoor plants,herbs,breads bakery,candy chocolate,snack food,meat seafood,baby food,dairy eggs and produce. \
The subcategory meat poultry includes jerky,seafood,sausages, exotic meats. \
The subcategory beverages includes juices,tea,coffee,soft drinks,powdered drink mixes,energy drinks,hot cocoa,coconut water,water,cocktail mixers,sports drinks. \
The subcategory gourmet gifts includes snack gifts,dessert gifts,tea gifts,cheese gifts,candy gifts,sauces gifts,seafood gifts,coffee gifts,chocolate gifts,fruit gifts,meat gifts,spices gifts. \
The subcategory sauces dips includes sauces,dips. \
The subcategory breakfast foods includes cereals,breakfast cereal bars,toaster pastries. \
The subcategory pantry staples includes sauces,condiments,dried beans,packaged meals side dishes,pasta noodles,oils,canned jarred food,jams. \
The subcategory cooking baking supplies includes chocolate,baking mixes,syrups,sugars sweeteners,nuts seeds,extracts flavoring,sugar substitutes,dessert toppings,flours meals,baking powder,food coloring,pastry decorations,coatings batters. \
The subcategory fresh flowers live indoor plants includes live indoor plants,fresh cut flowers. \
The subcategory herbs includes spices seasonings. \
The subcategory breads bakery includes cakes,cookies,pizza crusts,fresh baked cookies,tortillas,breakfast bakery,breadcrumbs,breads,packaged breads,stuffing,breadsticks. \
The subcategory candy chocolate includes chocolate assortments,licorice,chocolate bars,gum,gummy candies,gummy candy,chocolate truffles,hard candies,suckers lollipops,fudge,chewing gum,chocolate pretzels,bars,jelly beans,mints,taffy,halva,nut clusters,toffee,chocolate covered fruit,caramels,assortments,marshmallows,chocolate covered nuts. \
The subcategory snack food includes chips crisps,cookies,dried fruit raisins,party mix,jerky dried meats,crackers,pork rinds,trail mix,rice cakes,chips,puffed snacks,pretzels,popcorn,nuts seeds,granola bars,raisins,salsas,fruit leather,granola trail mix bars,pudding,dried fruit. \
The subcategory meat seafood includes wild game fowl,seafood,beef,chicken. \
The subcategory baby food includes baby formula,crackers biscuits,cereal,dinners,fruit. \
The subcategory dairy eggs includes cheese,milk,eggs,milk substitutes,ice cream frozen desserts. \
The subcategory produce includes fresh fruits,fresh vegetables. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."

amazon_531_food_simple_instruct_task="I want to perform hierarchical text classification, where the first level of the classification consists of 16 categories:  meat poultry,beverages,gourmet gifts,sauces dips,breakfast foods,pantry staples,cooking baking supplies,fresh flowers live indoor plants,herbs,breads bakery,candy chocolate,snack food,meat seafood,baby food,dairy milks and produce. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."

amazon_531_food_instruct_task_12="You are an expert in food product categorization. I have 16 subcategories for food product: \
The subcategory  meat poultry and seafood includes beef jerky, seafood like fish, and other meat \
The subcategory  beverages includes juices,tea,coffee,soft drinks,powdered drink mixes,energy drinks,hot cocoa,coconut water,water,cocktail mixers,sports drinks., \
The subcategory  gourmet gifts refers to edible products packaged in gift baskets and intended for use as presents. \
The subcategory  sauces dips refers to Chili sauce, garlic sauce, barbecue sauce, tomato sauce, and various dipping sauces. \
The subcategory  breakfast foods refer to food for breakfast like instant hot cereal ,oatmeal, breakfast cereal bars, breakfast cookies, Anything that mentions breakfast falls into this category. \
The subcategory  pantry staples includes sauces,condiments,dried beans,packaged meals side dishes,pasta noodles,oils,canned jarred food,jams. \
The subcategory  cooking baking supplies incuding baking mixes like scone mix, break mix, sugars sweeteners, nuts seeds, food coloring, sugar substitutes, flours. \
The subcategory  fresh flowers live indoor plants include flowers and live indoor plants. \
The subcategory  herbs spices seasonings includes sea salt, spice, ginger. \
The subcategory  breads bakery includes cakes, cookies, baked cookiers, breads, breadssticks. \
The subcategory  candy chocolate includes licorice, chocolate assortment, chocolate_bars, gum, gummy candy, hard candies, lollipops, mints, taffy , jelly beans. \
The subcategory  snack food includes chips crisps,cookies,party mix,jerky dried meats,crackers,pork rinds,trail mix,rice cakes,chips,puffed snacks,pretzels,popcorn,nuts seeds,granola bars,fruit leather,granola trail mix bars. \
The subcategory  baby food includes food for baby like baby formula, baby biscuits baby fruit.\
The subcategory  dairy milk  includes milk, cheese, eggs. \
The subcategory  produce includes fruits and vegetables. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."


amazon_531_toysgames_instruct_task="I want to perform hierarchical text classification, where the first level of the classification consists of 17 categories:  games,puzzles,arts crafts,action toy figures,dolls accessories,baby toddler toys,learning education,electronics for kids,stuffed animals plush,tricycles,grown up toys,dress up pretend play,novelty gag toys,building toys,sports outdoor play,hobbies,vehicles remote control. \
The subcategory games includes board games,card games,trading card games,travel games,battling tops,tile games,dvd games,stacking games,floor games,game collections,standard playing card decks,handheld games,dice gaming dice,game accessories,game room games. \
The subcategory puzzles includes jigsaw puzzles,d puzzles,pegged puzzles,puzzle accessories,floor puzzles,brain teasers,puzzle play mats. \
The subcategory arts crafts includes drawing painting supplies,drawing sketching tablets,clay dough,blackboards whiteboards,craft kits,easels,printing stamping,stickers,molding sculpting sticks,aprons smocks. \
The subcategory action toy figures includes figures,playsets,statues,accessories. \
The subcategory dolls accessories includes dolls,playsets,dollhouses,doll accessories,dollhouse accessories. \
The subcategory baby toddler toys includes shape sorters,push pull toys,music sound,crib toys attachments,stacking nesting toys,bath toys,rattles,activity play centers,car seat stroller toys,hammering pounding toys,rocking spring ride ons,balls,stuffed animals toys,mirrors,blocks,indoor climbers play structures,toy gift sets,teaching clocks. \
The subcategory learning education includes habitats,science,basic life skills toys,reading writing,geography,flash cards,mathematics counting,electronics. \
The subcategory electronics for kids includes electronic toys,music players karaoke,electronic pets,systems accessories,cameras camcorders,dance mats,personal video players accessories,plug play video games,walkie talkies. \
The subcategory stuffed animals plush includes animals figures,puppets,plush backpacks purses,plush pillows,teddy bears,plush puppets. \
The subcategory tricycles includes scooters wagons. \
The subcategory grown up toys includes die cast toy vehicles. \
The subcategory dress up pretend play includes pretend play,beauty fashion. \
The subcategory novelty gag toys includes toy balls,magic kits accessories,temporary tattoos,gag toys practical jokes,money banks,wind up toys,nesting dolls,miniatures,finger boards finger bikes,novelty spinning tops,viewfinders,prisms kaleidoscopes,flying toys. \
The subcategory building toys includes building sets,stacking blocks,marble runs. \
The subcategory sports outdoor play includes play tents tunnels,sports,inflatable bouncers,pools water fun,sandboxes accessories,gym sets swings,kites wind spinners,sand water tables,blasters foam play,gardening tools,pogo sticks hoppers,marble games,fitness equipment,ball pits accessories,beanbags foot bags,kickball playground balls,yo yos. \
The subcategory hobbies includes model building kits tools,rockets,radio control,scaled model vehicles,trains accessories,slot cars,coin collecting. \
The subcategory vehicles remote control includes play vehicles,play trains railway sets,vehicle playsets,die cast vehicles. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."
    
amazon_531_beauty_instruct_task="I want to perform hierarchical text classification, where the first level of the classification consists of 6 categories: makeup,skin care,bath body,tools accessories,hair care,fragrance. \
The subcategory makeup includes nails,face,body,eyes,makeup remover,lips,makeup sets. \
The subcategory skin care includes face,body,eyes,hands nails,sun,sets,maternity. \
The subcategory bath body includes cleansers,bath,scrubs body treatments,sets,bathing accessories. \
The subcategory tools accessories includes nail tools,cotton swabs,facial steamers,mirrors,makeup brushes tools,bags cases,hair coloring tools. \
The subcategory hair care includes styling products,shampoos,conditioners,hair color,styling tools,hair loss products,hair scalp treatments,shampoo conditioner sets,shampoo plus conditioner,hair relaxers,hair perms texturizers. \
The subcategory fragrance includes fragrance for women,fragrance for men,fragrance for children. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."

amazon_531_beauty_hair_instruct_task="You are an expert in e-commerce product categorization. I have a main category for hair care product with the following subcategories:  \
The subcategory hair styling products like pomade、styling gel, hair gel、groom gel, firm gel, bouncy creme，hair spray;  \
The subcategory shampoo are used for cleansing the scalp and hair, removing oil and dirt. such as dry shampoo, cream shampoo, gel shampoo and so on. \
The subcategory hair conditioner, \
The subcategory product related to hair color  like hair dye creams, coloring agents, and foam dyes that allow users to change their hair color with multiple shade options. \
The subcategory hair styling tools(some tools like  hair dryer ,curling or flat or styling iron, hair straightener,tangle tamer,  clipper, haircut kit, hair brush， hair razor, towel). \
The subcategory hair loss products includes hair growth serums, anti-hair loss shampoos, and toppik hair building fibers , couvre alopecia masking lotion , aimed at making hair grow , covering up the bald spots, improving hair loss and strengthening hair follicles. Suitable for individuals experiencing hair thinning or hereditary hair loss,  \
The subcategory hair scalp treatments includes hair masks, scalp serums, and nourishing oils designed to repair damaged hair and improve scalp health, promoting strong and h    ealthy hair growth. Ideal for users experiencing sensitive scalp, or severe hair damage.,  \
The subcategory shampoo conditioner sets(include shampoo and conditioner). \
The subcategory hair relaxers and perms texturizers. \
Please generate embeddings for the following product description to enhance the discernibility of the categories, making the differences more apparent during embedding."

amazon_531_beauty_hair_instruct_task_v3="You are an expert in e-commerce product categorization. I have a main category for hair care product with the following subcategories:  \
The subcategory hair styling products includes something for hair styling like Braid Maintenance, Creams, Gels & Lotions, Curl Enhancers, Hair Extensions & Wigs, Hair Sprays, Hair Styling Serums, Mousses & Foams, Pomades & Waxes.  \
The subcategory shampoo are used for cleansing the scalp and hair, removing oil and dirt. such as dry shampoo, cream shampoo, gel shampoo and so on. \
The subcategory hair conditioner. \
The subcategory hair color product focusing on products for color, such as  Chemical Hair Dyes, Color Correctors, Color Primers, Color Refreshers. \
The subcategory hair styling tools includes tools like  Braiders , Brushes, Combs, Diffusers, Hair Cutting Kits, Hair Dryers, Hair Rollers, Hot-Air Brushes, Irons, Scissors,Barrettes,Clips,Ponytail Holders,Hair Pins,Hair Drying Towels. \
The subcategory hair loss products includes hair growth serums, anti-hair loss shampoos, and toppik hair building fibers , couvre alopecia masking lotion , aimed at making hair grow , covering up the bald spots, improving hair loss and strengthening hair follicles. Suitable for individuals experiencing hair thinning or hereditary hair loss,  \
The subcategory hair scalp treatments includes hair masks, scalp serums, and nourishing oils designed to repair damaged hair and improve scalp health, promoting strong and h    ealthy hair growth. Ideal for users experiencing sensitive scalp, or severe hair damage.,  \
Please generate embeddings for the following product description to enhance the discernibility of the categories, making the differences more apparent during embedding."



amazon_531_beauty_instruct_task_v2="I want to perform hierarchical product classification, where the first level of the classification consists of 7 categories: makeup,skin care,bath body,tools accessories,hair care,fragrance. \
The subcategory makeup includes nails,face,body,eyes,makeup remover,lips,makeup sets. \
The subcategory skin care includes face care,body care,eyes care,hands and nails care, product related to sun protection or sun tanning, care sets, product for maternity. \
The subcategory bath body includes cleansers,bath,scrubs body treatments,sets,bathing accessories. \
The subcategory tools accessories includes nail tools,cotton swabs,facial steamers,mirrors,makeup brushes tools,bags cases,hair coloring tools. \
The subcategory hair care includes styling products, only shampoos, only conditioners,hair color product ,styling tools,hair loss related products,hair scalp treatments,shampoo and conditioner sets,shampoo conditioner 2 in 1 product,hair relaxers,hair perms texturizers. \
The subcategory fragrance includes women's fragrance,men's fragrance. \
Please generate embeddings for the following product to enhance the discernibility of the categories, making the differences more apparent during embedding."

amazon_531_healthpersonalcare_instruct_task="I want to perform hierarchical text classification, where the first level of the classification consists of 7 categories:   personal care,nutrition wellness,household supplies,health care,medical supplies equipment,baby child care,sexual wellness. \
The subcategory personal care includes deodorants antiperspirants,shaving hair removal,feminine care,oral hygiene,lip care products,eye care,foot care,body art,ear care. \
The subcategory nutrition wellness includes nutrition bars drinks,vitamins supplements,weight loss products,sports supplements. \
The subcategory household supplies includes household batteries,paper plastic,cleaning tools,laundry,household cleaning,lighters,dishwashing,air fresheners. \
The subcategory health care includes allergy,pill cases splitters,foot care,incontinence,cough cold,stress reduction,family planning contraceptives,pain relievers,first aid,diabetes,digestion nausea,thermometers,massage relaxation,thermometer accessories,smoking cessation,alternative medicine,women s health,therapeutic skin care,sleep snoring,stimulants. \
The subcategory medical supplies equipment includes daily living aids,tests,health monitors,braces,bathroom aids safety,occupational physical therapy aids,beds accessories,mobility aids equipment. \
The subclass baby child care will no longer be subdivided. \
The subcategory sexual wellness includes safer sex,adult toys games,sexual enhancers,sensual delights,bondage gear accessories,novelties,sex furniture. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."

amazon_531_babyproducts_instruct_task="I want to perform hierarchical text classification, where the first level of the classification consists of 12 categories:   gear,gifts,feeding,diapering,safety,nursery,bathing skin care,car seats accessories,strollers,pregnancy maternity,potty training,health baby care. \
The subcategory gear includes baby gyms playmats,backpacks carriers,activity centers entertainers,walkers,shopping cart covers,swings,playards,baby seats. \
The subcategory gifts includes albums,keepsakes,toy banks. \
The subcategory feeding includes bottle feeding,breastfeeding,pillows stools,food,highchairs booster seats,gift sets,solid feeding,pacifiers accessories. \
The subcategory diapering includes diaper changing kits,diaper pails refills,cloth diapers,diaper bags,wipes holders,disposable diapers,changing table pads covers,portable changing pads,diaper stackers caddies,cloth diaper accessories. \
The subcategory safety includesbathroom safety,gates doorways,monitors,edge corner guards,cabinet locks straps,rails rail guards,sleep positioners,kitchen safety,outdoor safety,electrical safety,harnesses leashes. \
The subcategory nursery includes furniture,nursery d cor,bedding. \
The subcategory bathing skin care includes skin care,bathing tubs seats,grooming healthcare kits,gift sets,soaps cleansers,non slip bath mats,washcloths towels,bubble bath. \
The subcategory car seats accessories includes car seats,accessories. \
The subcategory strollers includes accessories,joggers,standard,tandem,travel systems,lightweight,prams. \
The subcategory pregnancy maternity includes maternity pillows. \
The subcategory potty training includes potties seats,step stools,training pants,seat covers. \
The subcategory health baby care includes teethers,sun protection. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."

amazon_531_petsupplies_instruct_task="I want to perform hierarchical text classification, where the first level of the classification consists of 6 categories:  cats,dogs,fish aquatic pets,small animals,birds, bunny rabbit central. \
The subcategory cats includes litter housebreaking,food,toys,treats,beds furniture,health supplies,collars,educational repellents,feeding watering supplies,grooming,carriers strollers,cages,cat flaps. \
The subcategory dogs includes litter housebreaking,food,toys,treats,beds furniture,health supplies,collars,feeding watering supplies,training behavior aids,grooming,houses,carriers travel products,doors,apparel accessories. \
The subcategory fish aquatic pets includes food,health supplies,aquarium lights,pumps filters,test kits,water treatments,aquarium hoods,aquariums,automatic feeders,cleaners,aquarium d cor,aquarium starter kits,aquarium heaters,aquarium stands,hydrometers. \
The subcategory small animals such as  chickens, ferret, chinchillas, iguana, hamsters, guinea pigs, gerbils, and mouse, focus on the food and collars and house habitats for them. \
The subcategory birds includes food,toys,treats,health supplies,feeding watering supplies,cages accessories. \
The subcategory bunny rabbit central includes food,treats,collars,feeding watering supplies,houses habitats,rabbit hutches,carriers,odor stain removers. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding." 

amazon_531_petsupplies_instruct_task1="I want to perform the pet supplies product  classification,  the  classification consists of 6 categories: product for cats; product for dogs;product for  fish aquatic pets ; product for  small animals such as hamsters, guinea pigs, gerbils, and mice; product for birds, product for bunny rabbit . \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding." 


amazon_531_petsupplies_cat_instruct_task2="I want to perform product classification, where the classification consists of the following categories: \
    the subcategory cat litter box and  housebreaking. \
    the subcategory cat food including cat food, dog food, bird food, rabbit food , and other things for pet to eat. \
    the subcategory cat toys including balls and other toys. \
    the subcategory cat treats including pill pockets, chicken treat, dried treats, tasty treats for cats. \
    the subcategory cat beds furniture including fashion pet bed, tunnel,cat cave, kitty hoots, kitty pad, ramp. \
    the subcategory cat health supplies product, including renal essentials, calcium tablets, sprinkle capsules, balance supplemen which is good for pet's health\
    the subcategory cat collars product, including  safety cat collars, rhinestone cat collar,  kitten collar, control collar \
    the subcategory cat repellent devices like catstop, cat repellent spray, pet deterrent\
    the subcategory cat  feeding supplies and watering supplies, including food dispenser, pet food storage,feeder bowl, food, keeper, pet feeder, pet fountain, water bowl. \
    the subcategory cat grooming product, including includes pet shampoo and conditioners, furminator deshedding tool, grooming comb,  clipper kit, grooming brush \
    the subcategory carriers and strollers, includes pet carriers and pet strollers, pet trailer, pet cart. \
    the subcategory kennel crate cage, \
    the subcategory cat flaps including cat door, cat flaps, like petsafe door, panel, flaps, gate. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding." 

amazon_531_petsupplies_dog_instruct_task2="I want to perform pet supplies product classification, where the classification consists of the following categories: \
    the subcategory dog litter box and  housebreaking. \
    the subcategory dog food including cat food, dog food, bird food, rabbit food , and other things for pet to eat. \
    the subcategory dog toys including balls, \
    the subcategory dog treats, \
    the subcategory dog beds furniture including fashion pet bed, tunnel,cat cave, kitty hoots, kitty pad, ramp. \
    the subcategory dog health supplies,\
    the subcategory dog collars, \
    the subcategory dog educational repellents, \
    the subcategory dog  feeding supplies and watering supplies, including food dispenser, pet food storage,feeder bowl, food, keeper, pet feeder, pet fountain, water bowl. \
    the subcategory dog grooming, \
    the subcategory dog carriers strollers, \
    the subcategory dog cages, \
    the subcategory dog flaps including cat door, cat flaps, like petsafe door, panel, flaps, gate. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding." 


amazon_531_petsupplies_instruct_task2="I want to perform hierarchical text classification, where the classification consists of the following categories: \
litter box and  housebreaking, \
    the subcategory pet food including cat food, dog food, bird food, rabbit food , and other things for pet to eat. \
    the subcategory pet toys including balls, \
    the subcategory pet treats, \
    the subcategory pet beds furniture including fashion pet bed, tunnel,cat cave, kitty hoots, kitty pad, ramp. \
    the subcategory pet health supplies,\
    the subcategory pet collars, \
    the subcategory pet educational repellents, \
    the subcategory pet  feeding supplies and watering supplies, including food dispenser, pet food storage,feeder bowl, food, keeper, pet feeder, pet fountain, water bowl. \
    the subcategory grooming, \
    the subcategory cat carriers strollers, \
    the subcategory cages, \
    the subcategory cat flaps including cat door, cat flaps, like petsafe door, panel, flaps, gate. \
    the subcategory dog training behavior aids, \
    the subcategory dog houses, \
    the subcategory dog carriers travel products,\
    the subcategory dog doors, \
    the subcategory dog apparel accessories. \
    the subcategory fish aquatic pets aquarium lights,\
    the subcategory fish aquatic pets pumps filters,\
    the subcategory fish aquatic pets     test kits, \
    the subcategory fish aquatic pets  water treatments, \
    the subcategory fish aquatic pets aquarium hoods, \
    the subcategory     fish aquatic pets aquariums,\
    the subcategory         fish aquatic pets  automatic feeders, \
    the subcategory         fish aquatic pets cleaners, \
    the subcategory         fish aquatic pets aquarium d cor, \
    the subcategory         fish aquatic pets aquarium starter kits, \
     the subcategory        fish aquatic pets aquarium heaters, \
     the subcategory        fish aquatic pets aquarium stands, \
     the subcategory        fish aquatic pets hydrometers. \
     the subcategory        birds cages accessories. \
    the subcategory bunny rabbit central rabbit hutches, \
    the subcategory bunny rabbit carriers,\
    the subcategory bunny rabbit odor stain removers. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding." 


    
hwt_dataset_food_instruct_task="I want to perform hierarchical text classification, where the first level of the classification consists of 9 categories includes  Beverages, 	Cooking & Baking, 	Candy & Chocolate, 	Breakfast Foods, 	Baby Foods, 	Canned Dry & Packaged Foods, Snack Foods, 	Fresh Flowers & Live Indoor Plants, 	Gourmet Gifts . \
The subcategory Beverages includes  Coffee, 	Energy Drinks, 	Water, 	Tea, 	Protein Drinks, 	Powdered Drink Mixes, 	Soft Drinks, 	Sports Drinks, 	Hot Cocoa, 	Juices, 	Cocktail Mixers, 	Iced Tea & Lemonade, 	Coconut Water. \
The subcategory Cooking & Baking includes  Cooking Oils Vinegars & Sprays, 	Food Coloring, 	Frosting Icing & Decorations, 	Baking Mixes, 	Syrups Sugars & Sweeteners, 	Extracts & Flavoring, 	Pudding & Gelatin, 	Sugar Substitutes, 	Flours & Meals, 	Nuts & Seeds, 	Leaveners & Yeasts, 	Condensed & Powdered Milk, 	Baking Chocolates Carob & Cocoa, 	Breadcrumbs & Seasoned Coatings. \
The subcategory Candy & Chocolate includes  Chewing Gum, 	Hard Candy, 	Gummy Candy, 	Mints, 	Bars, 	Suckers & Lollipops, 	Assortments, 	Licorice, 	Jelly Beans. \
The subcategory Breakfast Foods includes  Breakfast & Cereal Bars, 	Cereals. \
The subcategory Baby Foods includes  Baby Formula, 	Fruit. \
The subcategory Canned Dry & Packaged Foods  includes  Sauces Gravies & Marinades, 	Jams Jellies & Sweet Spreads, 	Canned & Jarred Food, 	Dried Beans Grains & Rice, 	Condiments Pickles & Relishes, 	Herbs Spices & Seasonings, 	Pasta & Noodles, 	Oils Vinegars & Salad Dressings, 	Packaged Meals & Side Dishes. \
The subcategory Snack Foods includes  Jerky & Dried Meats, 	Salsas Dips & Spreads, 	Granola & Trail Mix Bars, 	Popcorn, 	Cookies, 	Puffed Snacks, 	Crackers, 	Chips & Crisps, 	Pretzels, 	Trail Mix, 	Party Mix. \
The subcategory Fresh Flowers & Live Indoor Plants includes  Live Indoor Plants. \
The subcategory Gourmet Gifts includes  Snack Gifts. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."



hwt_dataset_all_instruct_task1="Generate embeddings for the five kinds of products: Beauty; Grocery & Gourmet Food; Musical Instruments; Sports & Outdoors; Toys & Games, maximizing the distinguishability between different products."

hwt_dataset_all_instruct_task12="I want to perform hierarchical text classification, where the first level of the classification consists of 5 categories includes 'Beauty', 'Grocery & Gourmet Food', 'Musical Instruments',   'Sports & Outdoors', 'Toys & Games'.  \
The subcategory Beauty includes Skin Care, Tools & Accessories, Makeup, Hair Care, Bath & Body, Fragrance. \
The subcategory Grocery & Gourmet Food includes Beverages, Cooking & Baking, Candy & Chocolate, Breakfast Foods, Baby Foods, Canned  Dry & Packaged Foods, Snack Foods, Fresh Flowers & Live Indoor Plants. \
The subcategory Musical Instruments includes Drums & Percussion, Instrument Accessories, Keyboards & MIDI, Guitars, Ukuleles  Mandolins & Banjos, Wind & Woodwind Instruments, Studio Recording Equipment, Microphones & Accessories, Band & Orchestra,   Live Sound & Stage, Electronic Music  DJ & Karaoke, Amplifiers & Effects, Stringed Instruments, Bass Guitars. \
The subcategory Sports & Outdoors includes  Accessories, Hunting & Fishing, Outdoor Gear, Boating & Water Sports, Cycling, Exercise & Fitness, Fan Shop, Leisure Sports & Game Room, Action Sports, Team Sports, Snow Sports, Paintball & Airsoft, Golf, Racquet Sports, Equestrian Sports, Clothing,  Other Sports . \
The subcategory Toys & Games includes Puzzles, Learning & Education, Arts & Crafts, Games, Stuffed Animals & Plush, Novelty & Gag Toys, Dress Up & Pretend Play, Electronics for Kids, Toy Remote Control & Play Vehicles, Action Figures & Statues, Building Toys, Party Supplies, Dolls & Accessories, Grown-Up Toys,  Hobbies, Baby & Toddler Toys, Tricycles  Scooters & Wagons. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."
    
hwt_dataset_allclass59_instruct_task12="I want to perform hierarchical text classification, where the first level of the classification consists of 5 categories includes 'Beauty', 'Grocery & Gourmet Food', 'Musical Instruments',   'Sports & Outdoors', 'Toys & Games'.  \
The subcategory Beauty includes Skin Care, Tools & Accessories, Makeup, Hair Care, Bath & Body, Fragrance. \
The subcategory Grocery & Gourmet Food includes Beverages, Cooking & Baking, Candy & Chocolate, Breakfast Foods, Baby Foods, Canned  Dry & Packaged Foods, Snack Foods, Fresh Flowers & Live Indoor Plants. \
The subcategory Musical Instruments includes Drums & Percussion, Instrument Accessories, Keyboards & MIDI, Guitars, Ukuleles  Mandolins & Banjos, Studio Recording Equipment, Microphones & Accessories, Live Sound & Stage, Electronic Music  DJ & Karaoke, Amplifiers & Effects, Stringed Instruments, Bass Guitars. \
The subcategory Sports & Outdoors includes  Accessories, Hunting & Fishing, Outdoor Gear, Boating & Water Sports, Cycling, Exercise & Fitness, Fan Shop, Leisure Sports & Game Room, Action Sports, Team Sports, Snow Sports, Paintball & Airsoft, Golf, Racquet Sports, Equestrian Sports, Clothing,  Other Sports . \
The subcategory Toys & Games includes Puzzles, Learning & Education, Arts & Crafts, Games, Stuffed Animals & Plush, Novelty & Gag Toys,  Electronics for Kids, Toy Remote Control & Play Vehicles, Action Figures & Statues, Building Toys, Party Supplies, Dolls & Accessories, Grown-Up Toys,  Hobbies, Baby & Toddler Toys, Tricycles  Scooters & Wagons. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."
    
hwt_dataset_265_instruct_task12="I want to perform hierarchical text classification, where the first level of the classification consists of 5 categories includes 'Beauty', 'Grocery & Gourmet Food', 'Musical Instruments',   'Sports & Outdoors', 'Toys & Games'.  \
The subcategory Beauty includes Skin Care, Tools & Accessories, Makeup, Hair Care, Bath & Body, Fragrance. \
The subcategory Grocery & Gourmet Food includes Beverages, Cooking & Baking, Candy & Chocolate, Breakfast Foods, Baby Foods, Canned  Dry & Packaged Foods, Snack Foods, Fresh Flowers & Live Indoor Plants. \
The subcategory Musical Instruments includes Drums & Percussion, Instrument Accessories, Keyboards & MIDI, Guitars, Ukuleles  Mandolins & Banjos, Studio Recording Equipment, Microphones & Accessories, Electronic Music  DJ & Karaoke, Amplifiers & Effects. \
The subcategory Sports & Outdoors includes  Accessories, Hunting & Fishing, Outdoor Gear, Boating & Water Sports, Exercise & Fitness, Fan Shop, Leisure Sports & Game Room, Action Sports, Team Sports, Snow Sports, Paintball & Airsoft, Golf, Racquet Sports, Equestrian Sports,  Other Sports . \
The subcategory Toys & Games includes Puzzles, Learning & Education, Arts & Crafts, Games, Stuffed Animals & Plush, Novelty & Gag Toys, Electronics for Kids,  Action Figures & Statues, Building Toys, Party Supplies, Dolls & Accessories, Grown-Up Toys,  Hobbies, Tricycles  Scooters & Wagons. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."

    
# task = "我想做层次化文本分类，分类体系第一级有6个类别，分别为grocery gourmet food、toys games、beauty、health personal care、baby products和pet supplies。其中和nutrition wellness相关的内容如nutrition bars drinks、vitamins supplements、weight loss products、sports supplements这类的商品应该归属于health personal care类别。请为下面的文本生成embedding，以便增强类别的辨识度，更明显地体现分类时的差异"

amazon_task_en = "I want to perform hierarchical text classification, where the first level of the classification system consists of 6 categories: grocery gourmet food, toys games, beauty, health personal care, baby products, and pet supplies. Content related to nutrition wellness, such as nutrition bars drinks, vitamins supplements, weight loss products, and sports supplements, should be categorized under the health personal care category. Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."

wos_task_en = "I want to perform hierarchical text classification, where the first level of the classification system consists of 7 categories: 'Computer  Science', 'Medical  Science,', 'Civil  Engineering', 'Electrical  Engineering', 'biochemistry', 'Mechanical  Engineering' and 'Psychology'. You shold know the differece between Civil  Engineering and Mechanical  Engineering. The category 'Mechanical  Engineering' includes 'Fluid mechanics', 'Hydraulics', 'computer-aided design', 'Manufacturing engineering', 'Machine design', 'Thermodynamics', 'Materials Engineering', 'Strength of materials', 'Internal combustion engine'.  The category 'Civil Engineering' includes:'Green Building', 'Water Pollution', 'Smart Material', 'Ambient Intelligence', 'Construction Management', 'Suspension Bridge', 'Geotextile', 'Stealth Technology', 'Solar Energy', 'Remote Sensing', 'Rainwater Harvesting', 'Transparent Concrete', 'Highway Network System', 'Nano Concrete', 'Bamboo as a Building Material', 'Underwater Windmill'.  Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."


dbpedia_task_en = "I want to perform hierarchical text classification, where the first level of the classification system consists of 7 categories: agent,work,place,species,unit of work,event,sports season,device,topical concept. You should know that Class agent includes:politician,organisation,person,athlete,motorcycle rider,company,winter sport player,educational institution,actor,sports manager,cleric,boxer,artist,coach,comics character,group,broadcaster,sports team,scientist,british royalty,fictional character,presenter,writer,sports league,gridiron football player,racing driver,wrestler,organisation member,musical artist,volleyball player. Class work includes musical work,periodical literature,cartoon,software,comic,written work,song,database. Class place includes natural place,building,settlement,route of transportation,clerical administrative region,body of water,infrastructure,venue,tower,sport facility,celestial body,satellite,amusement park attraction,race track,stream , station. Class species includes animal,plant,eukaryote,horse,flowering plant. class unit of work includes legal case. Class event includes societal event,tournament,olympics,race,sports event,natural event. Class sports season includes football league season,sports team season. Class device includes engine. Class topical concept includes genre.  Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."







wos5736_instruct_task12 = "I want to perform hierarchical text classification for science papers, the first level of the classification system consists of 3 categories: 'Electrical  Engineering', 'biochemistry',  'Psychology'. \
The category 'Electrical  Engineering' includes Electricity, Digital control, Operational amplifier.  \
The category 'biochemistry' includes Molecular biology, Immunology, Polymerase chain reaction, Northern blotting. \
The category 'Psychology' includes Social cognition, Child abuse, Depression , Attention.\
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."

wos5736_instruct_task2 = "I want to perform hierarchical text classification for science papers, including 11 categories: \
'Electricity in Electrical  Engineering', \
'Digital control in Electrical  Engineering', \
'Operational amplifier in Electrical  Engineering',\
'Molecular biology of biochemistry',  \
'Immunology of biochemistry', \
' Polymerase chain reaction of biochemistry', \
'Northern blotting of biochemistry', \
' Social cognition of Psychology'. \
'Child abuse of Psychology', \
'Depression of Psychology', \
'Attention of Psychology',\
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."


wos_instruct_task12 = "I want to perform hierarchical text classification for science papers, the first level of the classification system consists of 7 categories: Computer  Science, Electrical  Engineering,  Psychology,  Mechanical  Engineering,Civil  Engineering,  Medical  Science,  biochemistry. \
The subcategory for 'Computer Science' includes : Computer vision,Distributed computing,Structured Storage,Symbolic computation,Algorithm design,Computer programming,Data structures,Bioinformatics,Machine learning,network security,Cryptography,Operating systems,Computer graphics,Image processing,Parallel computing,Relational databases,Software engineering.  \
The subcategory for Electrical  Engineering includes : Analog signal processing, Control engineering, Digital control, Electrical circuits,  Electrical network, Electricity, Electric motor, Lorentz force law, Microcontroller, Operational amplifier, PID controller, Satellite radio, Signal-flow graph,  State space representation, System identification, Voltage law.\
The subcategory for Psychology includes : Antisocial personality disorder, Attention, Borderline personality disorder, Child abuse, Depression, Eating disorders, False memories, Gender roles, Leadership, Media violence, Nonverbal communication, Person perception, Prejudice, Prenatal development, Problem-solving, Prosocial behavior, Schizophrenia, Seasonal affective disorder, Social cognition.\
The subcategory for   Mechanical  Engineering  includes: computer-aided design, Fluid mechanics, Hydraulics, Internal combustion engine, Machine design, Manufacturing engineering, Materials Engineering, Strength of materials, Thermodynamics.\
The subcategory for  Civil  Engineering includes: Ambient Intelligence, Construction Management, Geotextile, Green Building,  Rainwater Harvesting, Remote Sensing, Smart Material, Solar Energy, Stealth Technology, Suspension Bridge,  Water Pollution,.\
The subcategory for Medical  Science  includes : Addiction, Allergies, Alzheimer's Disease, Ankylosing Spondylitis, Anxiety, Asthma, Atopic Dermatitis, Atrial Fibrillation, Autism, Bipolar Disorder, Birth Control, Cancer, Children's Health, Crohn's Disease, Dementia, Diabetes, Digestive Health, Emergency Contraception, Fungal Infection, Headache, Healthy Sleep, Heart Disease, Hepatitis C, Hereditary Angioedema, HIV/AIDS, Hypothyroidism, Idiopathic Pulmonary Fibrosis, Irritable Bowel Syndrome, Kidney Health, Low Testosterone, Lymphoma, Medicare, Menopause, Mental Health, Migraine, Multiple Sclerosis, Myelofibrosis, Osteoarthritis, Osteoporosis,  Overactive Bladder, Parenting, Parkinson's Disease, Polycythemia Vera, Psoriasis, Psoriatic Arthritis, Rheumatoid Arthritis, Senior Health, Skin Care, Smoking Cessation, Sports Injuries, Sprains and Strains, Stress Management, Weight Loss.\
The subcategory for  biochemistry includes: Cell biology, Enzymology, Genetics, Human Metabolism, Immunology, Molecular biology, Northern blotting, Polymerase chain reaction, Southern blotting.\
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."

wos_instruct_task1 = "I want to perform hierarchical text classification for science papers, the first level of the classification system consists of 7 categories: Computer  Science, Electrical  Engineering,  Psychology,  Mechanical  Engineering,Civil  Engineering,  Medical  Science,  biochemistry. \
Please generate embeddings for the following paper abstract to enhance the discernibility of the categories, making the differences more apparent during embedding."


wos11967_instruct_task12 = "I want to perform hierarchical text classification for science papers, the first level of the classification system consists of 7 categories: Computer  Science, Electrical  Engineering,  Psychology,  Mechanical  Engineering,Civil  Engineering,  Medical  Science,  biochemistry. \
The subcategory for Computer Science includes : Computer vision, Machine learning, network security, Cryptography, Operating systems.  \
The subcategory for Electrical  Engineering includes : Electricity, Digital control, Electrical circuits.\
The subcategory for Psychology includes : Prejudice, Social cognition, Person perception, Nonverbal communication, Prosocial behavior. \
The subcategory for Mechanical  Engineering  includes: computer-aided design, Hydraulics,  Manufacturing engineering, Machine design, Fluid mechanics. \
The subcategory for Civil  Engineering includes: Ambient Intelligence, Geotextile, Remote Sensing, Rainwater Harvesting,  Water Pollution. \
The subcategory for Medical  Science  includes : Addiction, Allergies, Alzheimer's Disease, Ankylosing Spondylitis, Anxiety.\
The subcategory for  biochemistry includes: Molecular biology, Cell biology, Human Metabolism, Immunology, Genetics. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."


wos11967_instruct_task12_v2 = "I want to perform hierarchical text classification for science papers, the first level of the classification system consists of 7 categories: Computer  Science, Electrical  Engineering,  Psychology,  Mechanical  Engineering,Civil  Engineering,  Medical  Science,  biochemistry. \
The subcategory for Computer Science includes : Computer vision, Machine learning, network security, Cryptography, Operating systems.  \
The subcategory for Electrical  Engineering includes : Electricity, Digital control, Electrical circuits.\
The subcategory for Psychology includes : Prejudice, Social cognition, Person perception, Nonverbal communication, Prosocial behavior. \
The subcategory for Mechanical  Engineering  includes: computer-aided design, Hydraulics,  Manufacturing engineering, Machine design, Fluid mechanics. \
The subcategory for Civil  Engineering includes: Ambient Intelligence, Geotextile, Remote Sensing, Rainwater Harvesting,  Water Pollution. \
The subcategory for Medical  Science  includes : drug or internet or alcohol or substance Addiction, Allergies, Alzheimer's Disease, Ankylosing Spondylitis, Anxiety and depression.\
The subcategory for  biochemistry includes: Molecular biology, Cell biology, Human Metabolism, Immunology & Immunotherapy, Genetics. \
Note that  allergies, as well as those related to drugs, foods, or other addictions,  anxiety and depression should fall under the domain of medical science. \
Note that Machines related to electronics such as electrical machines, electric machines, and permanent magnet machines all fall under the category of machine design in Mechanical  Engineering. and Any topic related to hydraulic such as hydraulic model, hydraulic structures ,  hydraulic design, fluid power, hydraulic control or performance or parameters, etc., should be classified under Mechanical Engineering. \
Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."




dbpedia_task0 ="Generate embeddings for these wikipedia articles with different topic, maximizing the distinguishability between different topic."
dbpedia_task1 = "I want to conduct hierarchical topic classification, The first-level classification is divided into 9 themes: agent, work, place, species, unit of work, event, sports season, device, topical concept.\
Please generate embeddings for the following article to enhance the discernibility of the categories, making the differences more apparent during embedding." 
dbpedia_task12 = "I want to perform hierarchical text classification, where the first level of the classification system consists of 9 categories: agent,work,place,species,unit of work,event,sports season,device,topical concept. \
The subcategory of agent includes:politician,organisation,person,athlete,motorcycle rider,company,winter sport player,educational institution,actor,sports manager,cleric,boxer,artist,coach,comics character,group,broadcaster,sports team,scientist,british royalty,fictional character,presenter,writer,sports league,gridiron football player,racing driver,wrestler,organisation member,musical artist,volleyball player. Class work includes musical work,periodical literature,cartoon,software,comic,written work,song,database. Class place includes natural place,building,settlement,route of transportation,clerical administrative region,body of water,infrastructure,venue,tower,sport facility,celestial body,satellite,amusement park attraction,race track,stream , station. Class species includes animal,plant,eukaryote,horse,flowering plant. class unit of work includes legal case. Class event includes societal event,tournament,olympics,race,sports event,natural event. Class sports season includes football league season,sports team season. Class device includes engine. Class topical concept includes genre.  Please generate embeddings for the following text to enhance the discernibility of the categories, making the differences more apparent during embedding."
