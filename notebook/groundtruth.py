"""Ground-truth registry generated from the GORE annotation workbook.

Source workbook: C:/Users/agnes/Desktop/tesi originale/LLM_MultiAgentChain_GoalOrientedAPIAlignement/notebook/GORE_groundtruth_annotation.xlsx
Regenerate with: python notebook/update_groundtruth_from_excel.py
"""

# This file is generated. Edit the Excel workbook, then regenerate this module.

SIA_PROJECT_25_26 = {   'name': 'SIA Project 25 26',
    'description': 'In the Municipality of Turin, a system called Participium is being developed — '
                   'an information system that enables citizen participation in the management of '
                   'urban environments. It allows citizens to interact with the public '
                   'administration by reporting inconveniences and malfunctions found in the city '
                   '(e.g., potholes in the asphalt, sidewalks with architectural barriers, garbage '
                   'in the streets, broken streetlights, etc.). An example of a similar '
                   'information system is IRIS in Venice: '
                   '[https://iris.sad.ve.it/](https://iris.sad.ve.it/?utm_source=gemini). The '
                   'system, once developed, will be made available as open-source software to all '
                   'Italian public administrations through the portal '
                   '[https://developers.italia.it/](https://developers.italia.it/?utm_source=gemini).\n'
                   '\n'
                   'A. REPORTS\n'
                   'Citizens can submit reports only if they have registered in the system with a '
                   'username, first name, and last name. Once the registration is received, the '
                   'user gets an email with a confirmation link. The registration becomes valid, '
                   'and the user can use the system only after confirming through that link. Once '
                   'registered, a citizen can submit reports by selecting a point on the map of '
                   'Turin (which will be saved with latitude and longitude values) and filling in '
                   'a problem form with the following mandatory fields: title, textual '
                   'description, category (chosen from a predefined list). It is also mandatory to '
                   'attach one or more photos (up to 3 per report, with each photo stored with its '
                   'file path on the server). The possible problem categories are:\n'
                   '• Waterworks – Drinking water\n'
                   '• Architectural Barriers\n'
                   '• Sewerage\n'
                   '• Public Lighting\n'
                   '• Waste\n'
                   '• Road Signs and Traffic Lights\n'
                   '• Roads and Urban Furniture\n'
                   '• Public Green Areas and Playgrounds\n'
                   '• Other\n'
                   'After entering all the information and the pictures, the system asks the '
                   'citizen whether they want the report to be anonymous (the name will not appear '
                   'in the public list of all reports).\n'
                   '\n'
                   'B. REPORT LIFECYCLE\n'
                   'Once submitted, the report is in the “Pending Approval” state until the '
                   'Organization Office of the Municipality of Turin performs a preliminary review '
                   'of citizen reports, marking them as accepted or rejected. The possible states '
                   'for a report are: Pending Approval, In Progress, Work in Progress, Suspended, '
                   'Rejected. After approval, accepted reports move to the “In Progress” state, at '
                   'which point they are assigned to the technical office responsible, based on '
                   'the problem category. Once the intervention is planned, the state changes to '
                   '“Work in Progress”, indicating that the issue resolution has started. In some '
                   'cases, for organizational or technical reasons, the report can be set to '
                   '“Suspended”, awaiting further evaluation or resources. When the problem is '
                   'solved, the technical office updates the status and closes the report. In case '
                   'of rejection, a written explanation from the Organization Office is mandatory '
                   '(see next section).\n'
                   'If the intervention must be carried out by maintenance personnel from an '
                   'external company (for example, Enel X for Public Lighting, or company Y for '
                   'specific reports based on their content), two cases are possible:\n'
                   '\n'
                   '* Case 1: The company has access to Participium. In this case, the technical '
                   'office assigns the report to users from the corresponding company. External '
                   'maintenance personnel can move the report from Assigned to In Progress. Staff '
                   'from the technical office and external maintenance workers can exchange '
                   'information and comments through the report, but these are not visible to the '
                   'reporting citizen (nor to other citizens). Once the work is completed, the '
                   'external maintainer can mark the report as resolved.\n'
                   'Note: Automatic assignment of all reports in a certain category to the '
                   'external company (bypassing the initial review by the public relations officer '
                   'of the municipality) is possible only if previously configured by the '
                   'municipal administrator.\n'
                   '* Case 2: The company does NOT have access to Participium. In this case, the '
                   'external company updates the technical office outside Participium, and once '
                   'the problems are solved, a staff member from the technical office will '
                   'manually move the report to the Resolved state.\n'
                   '\n'
                   'C. CITIZEN UPDATES\n'
                   'To strengthen trust between citizens and institutions, the citizen can receive '
                   'updates regarding their own reports through various channels. First, at every '
                   'state change, the citizen receives a notification on the platform with the '
                   'corresponding update. Furthermore, municipal operators updating reports can '
                   'send the reporting citizen a message through the platform, to which it can '
                   'reply. The system must allow this functionality to be accessible by external '
                   'chatbots as well. Each time the citizen receives a notification on the '
                   'platform, they also get an email (this option can be disabled in the user '
                   'settings panel, where they can also upload a personal photo).\n'
                   'Moreover, after the approval phase, accepted reports immediately become '
                   'visible on the Participium portal: they appear both on an interactive map of '
                   'Turin, geolocated based on the citizen’s selected point, and in a summary '
                   'table that allows filtering and sorting reports by category, status, or date. '
                   'In both views, the reporter’s name (“anonymous” if that option was chosen) and '
                   'the report title are displayed. Clicking on the title opens a page with the '
                   'description.\n'
                   'Even non-registered users can view both the maps and the summary table. Once '
                   'logged into Participium, citizens can follow other citizens’ open reports and '
                   'receive notifications (following the same rules as for their own reports).\n'
                   '\n'
                   'D. STATISTICS\n'
                   'The system allows viewing public and private statistics. Public statistics are '
                   'visible in a dedicated section of the website and concern the number of '
                   'reports per category, and trends over day, week, or month. They are also '
                   'visible to non-registered users. In the private section, accessible only to '
                   'administrators, it is possible to view, in addition to the public statistics, '
                   'charts and tables.',
    'actors': [   'Visitor',
                  'Citizen',
                  'Technical Office Staff',
                  'Organization Office Staff',
                  'Municipal Administrator',
                  'External Maintenance Staff'],
    'highLevelGoals': [   'View public statistics',
                          'Register an account',
                          'Report urban issues',
                          'Exchange messages',
                          'Manage report lifecycle',
                          'View private statistics',
                          'Configure automatic report assignment',
                          'Manage account information',
                          'Browse public reports',
                          'Receive updates about reports'],
    'lowLevelGoals': [   'Activate account',
                         'Submit account registration',
                         'Select the report location on the map',
                         'Select anonymity for a report',
                         'Submit a report',
                         'Send a message to the reporting citizen',
                         'Activate email notifications',
                         'Deactivate email notifications',
                         'Upload a profile photo',
                         'Enable or disable automatic assignment of a category to an external '
                         'company',
                         'Approve a report',
                         'Reject a report specifying the rejection reason',
                         'Add an internal comment to a report',
                         'Assign a report to an external company',
                         'Follow a report',
                         'View the details of a report',
                         'Log in',
                         'Attach photos to a report',
                         'Set a report status to In Progress',
                         'Set a report status to Work in Progress',
                         'Suspend a report',
                         'Mark a report as resolved',
                         'Receive a platform notification when the status of an own or followed '
                         'report changes',
                         'Receive an email notification when the status of an own or followed '
                         'report changes',
                         'Reply to a message from a municipal operator',
                         'View accepted reports on the interactive map',
                         'View accepted reports in the summary table',
                         'Filter reports by category, status, or date',
                         'Sort reports by category, status, or date',
                         'View public report statistics',
                         'View private report statistics']}

SIA_PROJECT_24_25 = {   'name': 'Assegno Unico Universale - SIA Project 24 25',
    'description': 'The system supports the online management of applications for the Italian '
                   'Assegno Unico Universale (AUU), a financial benefit provided by INPS to '
                   'families with dependent children. Authenticated applicants can create '
                   'applications, specify child, household, custody and payment information, save '
                   'drafts, resume applications, accept declarations and privacy notices, and '
                   'submit applications. The system validates fiscal codes, handles ISEE '
                   'information, calculates payments, and allows INPS employees to accept, reject '
                   'or suspend applications. Suspended applications may require additional '
                   'documents. Another parent can complete an existing application and modify '
                   'allowance distribution information. Applicants can also add children to '
                   'existing applications, consult submitted applications, access general AUU '
                   'information, and simulate the expected monthly allowance.',
    'actors': ['Applicant', 'Other Parent', 'INPS Employee'],
    'highLevelGoals': [   'Submit and manage an AUU application',
                          'Complete an AUU application started by another parent',
                          'Evaluate and manage submitted AUU applications',
                          'Access AUU information and decision-support services'],
    'lowLevelGoals': [   'Specify personal and dependent child information',
                         'Specify custody and household information',
                         'Specify the requested allowance distribution',
                         'Specify applicant payment information',
                         "Specify the other parent's payment information",
                         'Specify information about adult dependent children',
                         'Review application data before submission',
                         'Submit the AUU application',
                         'Download the application receipt',
                         'Save an incomplete application as a draft',
                         'Resume and complete a saved application',
                         'Consult an AUU application',
                         'Upload additional documents requested by INPS',
                         'Add a child to an existing application',
                         'Consult general information about the AUU',
                         'Simulate the monthly AUU amount',
                         'Confirm or modify the requested allowance distribution',
                         'Specify payment information when completing an application started by '
                         'another parent',
                         'Accept an AUU application',
                         'Reject an AUU application',
                         'Suspend an AUU application',
                         'Accept the declarations of responsibility and the GDPR privacy notice',
                         'Correct invalid application information',
                         'Interrupt an application without saving it',
                         'Discard a saved draft and restart the application',
                         'Specify the additional documents required from the applicant',
                         'Reconfirm application data after a suspension',
                         'Log in with a digital identity']}

SIA_PROJECT_23_24 = {   'name': 'La Reine Marlene - SIA Project 23 24',
    'description': 'La Reine Marlene is a company specialized in the sale of home materials and '
                   'products. The information system manages relationships with suppliers, product '
                   'proposals, supplier orders, incoming goods, inventory and replenishment, '
                   'physical and online purchases, deliveries, customer support, and business '
                   'statistics. Suppliers submit proposals that are evaluated by the '
                   'administration department. Goods are verified when they arrive at stores. '
                   'Inventory is monitored and automatic replenishment can be configured. '
                   'Customers can purchase products online, pay through multiple methods, and '
                   'track deliveries. Support requests are managed through tickets with automatic '
                   'assignment, response deadlines and reassignment mechanisms. Company management '
                   'can analyze data concerning orders, sales, inventory and customer support.',
    'actors': [   'Supplier',
                  'Administration Department',
                  'Goods Receiving Staff',
                  'Order Department',
                  'Customer',
                  'Customer Service Agent',
                  'Company Management',
                  'Store Staff'],
    'highLevelGoals': [   'Supply products to La Reine Marlene',
                          'Manage product proposals',
                          'Manage incoming goods',
                          'Manage supplier orders',
                          'Purchase products',
                          'Request customer support',
                          'Provide customer support',
                          'Analyze company performance',
                          'Manage product inventory and replenishment',
                          'Manage online order delivery'],
    'lowLevelGoals': [   'Submit a product proposal',
                         'Accept a product proposal',
                         'Request revisions to a product proposal with an explanatory comment',
                         'Reject a product proposal',
                         'Prepare a supplier order',
                         'Send a supplier order',
                         'Confirm an automatically generated order',
                         'Verify delivered goods',
                         'Report problems with delivered products',
                         'Select a store location for accepted products',
                         'Mark a supplier order as completed',
                         'Update an incomplete supplier order',
                         'Pay a supplier order',
                         'Configure automatic product reordering',
                         'Monitor product stock levels',
                         'Browse the product catalogue',
                         'Select products and quantities',
                         'Place an online order',
                         'Pay an online order',
                         'Track shipment status',
                         'Open a customer support ticket',
                         'View responses to a support ticket',
                         'Respond to a customer support ticket',
                         'View business statistics',
                         'Register as a supplier',
                         'Set the retail price for an accepted product proposal',
                         'Modify and resubmit a product proposal',
                         'Confirm accepted products for admission into the store',
                         'Print the price of an accepted product',
                         'Reply to a customer service response',
                         'Close the verification of a delivered order',
                         'Register as a customer']}

SIA_PROJECT_22_23 = {   'name': 'Ethical Purchasing Group - SIA Project 22 23',
    'description': 'The system supports the management of an Ethical Purchasing Group associated '
                   'with a km-zero fruit and vegetable store. Farmers periodically enter product '
                   'availability, customers browse offers and place orders, and store employees '
                   'may enter orders on behalf of customers. The system manages availability '
                   'constraints, priority-based allocation, automatic payment through customer '
                   'wallets, pickup reservations, farmer deliveries, reminders and refunds. It '
                   'also maintains a fair-play score for customers, which influences priority when '
                   'available product quantities are insufficient.',
    'actors': ['Customer', 'Farmer', 'Store Employee'],
    'highLevelGoals': [   'Manage customer orders',
                          'Manage product supply and provisioning',
                          'Manage customer payments',
                          'Manage order pickup'],
    'lowLevelGoals': [   'Place an order',
                         'Accept a modified order',
                         'Book an order pickup',
                         'Enter an order on behalf of a customer',
                         'View order details',
                         'Confirm an order pickup',
                         'Recharge a customer wallet',
                         'Confirm product delivery',
                         'Enter product availability',
                         'Confirm product availability',
                         'Browse offers from farmers',
                         'Select products and quantities for an order',
                         'Modify an order that does not meet minimum quantity constraints',
                         'View the order pickup schedule']}

SIA_PROJECT_21_22 = {   'name': 'Event Organization Portal - SIA Project 21 22',
    'description': 'The system is a web portal designed to support people who want to organize\n'
                   '    an event by bringing together, in a single platform, information about '
                   'event\n'
                   '    guests and the various services required for the event.\n'
                   '\n'
                   '    The platform supports three main types of users: event organizers, '
                   'service\n'
                   '    providers, and invited guests.\n'
                   '\n'
                   '    Service providers register on the platform by providing authentication\n'
                   '    credentials, personal information, and company details, including VAT\n'
                   '    number, company address, and accepted payment methods.\n'
                   '\n'
                   '    Once registered, providers can publish the services offered by their '
                   'company.\n'
                   '    Each service belongs to a category, such as catering, hairdressing, '
                   'florist\n'
                   '    services, or transportation. For each service, the provider can configure '
                   'a\n'
                   '    virtual showcase containing demonstration photos, prices, and the time '
                   'slots\n'
                   '    during which the service is available.\n'
                   '\n'
                   '    Event organizers register by providing credentials, personal information,\n'
                   '    residence information, their fiscal code, and an IBAN that can be used '
                   'for\n'
                   '    service payments.\n'
                   '\n'
                   '    Through the portal, organizers can create an event by specifying the '
                   'event\n'
                   '    type, date, location, and the planned budget. During event creation, they '
                   'can\n'
                   '    also select from a predefined list the categories of services for which '
                   'they\n'
                   '    want to receive support.\n'
                   '\n'
                   '    Organizers can add invited guests to the event by entering their personal\n'
                   '    information, email address, and guest category, such as relative, friend, '
                   'or\n'
                   '    colleague. For each invited guest, the system generates an invitation '
                   'code\n'
                   '    that is sent by email and is used by the guest as a password to access '
                   'the\n'
                   '    platform.\n'
                   '\n'
                   '    Each event has an associated blog where both organizers and invited '
                   'guests\n'
                   '    can publish information about the event, upload photos or videos, create '
                   'new\n'
                   '    posts, and comment on existing posts.\n'
                   '\n'
                   '    After creating an event, the organizer can browse the service showcases\n'
                   '    provided by suppliers for the selected service categories. The organizer '
                   'can\n'
                   '    then request an appointment with a provider by choosing a time compatible\n'
                   "    with the provider's availability and specifying appointment details and "
                   'any\n'
                   '    special requests.\n'
                   '\n'
                   '    Providers can view appointment requests through their interface and '
                   'decide\n'
                   '    whether to approve them. Approved appointments are automatically '
                   'displayed\n'
                   '    in a calendar associated with the event, which the organizer can consult '
                   'to\n'
                   '    view all confirmed service appointments.\n'
                   '\n'
                   '    After an appointment between the organizer and the provider has taken '
                   'place,\n'
                   '    the provider prepares a quotation for the agreed service and uploads the\n'
                   '    corresponding amount to the portal, associating it with the completed\n'
                   '    appointment.\n'
                   '\n'
                   '    Once a quotation has been uploaded, the organizer can accept or reject '
                   'it.\n'
                   '    If the quotation is accepted, its total amount is deducted from the '
                   'remaining\n'
                   '    event budget.\n'
                   '\n'
                   '    When accepting a quotation, the organizer selects both the payment '
                   'schedule,\n'
                   '    which can consist of installments, deposit plus final balance, or full '
                   'payment,\n'
                   '    and the payment method, which can be online or on-site.\n'
                   '\n'
                   '    If payment is made on-site, the provider confirms the received amount '
                   'through\n'
                   '    the portal. If payment is divided into multiple installments, the '
                   'provider\n'
                   '    confirms each received installment separately.\n'
                   '\n'
                   '    At any time, the organizer can consult the history of contracts '
                   'associated\n'
                   '    with the event, including their status and cost, and can also view the\n'
                   '    remaining event budget.',
    'actors': ['Organizer', 'Service Provider', 'Guest'],
    'highLevelGoals': [   'Manage event services',
                          'Manage an event',
                          'Manage the event blog',
                          'Access the platform'],
    'lowLevelGoals': [   'Register as a service provider',
                         'Add offered services',
                         'Confirm an on-site payment',
                         'Register as an organizer',
                         'Create an event',
                         'Add invited guests',
                         'Select the service types needed for an event',
                         'View appointment calendar',
                         'View the remaining event budget',
                         'View contract history',
                         'Create a blog post as an organizer',
                         'Create a blog post as a guest',
                         'Upload demonstration photos for a service',
                         'Specify the price of a service',
                         'Specify service availability time slots',
                         'Browse service showcases',
                         'Request an appointment with a service provider',
                         'Approve an appointment request',
                         'Upload a quotation for a completed appointment',
                         'Accept a quotation',
                         'Reject a quotation',
                         'Select a payment schedule',
                         'Select a payment method',
                         'Access the platform using an invitation code',
                         'Upload photos or videos to the event blog',
                         'Comment on an event blog post',
                         'View appointment requests',
                         'Log in']}

SIA_PROJECT_GENOME = {   'name': 'Genome Nexus',
    'description': 'Genome Nexus, a comprehensive one-stop resource for fast, automated and '
                   'high-throughput annotation and interpretation of genetic variants in cancer. '
                   'Genome Nexus integrates information from a variety of existing resources, '
                   'including databases that convert DNA changes to protein changes, predict the '
                   'functional effects of protein mutations, and contain information about '
                   'mutation frequencies, gene function, variant effects, and clinical '
                   'actionability.',
    'link-readme': 'https://github.com/WebFuzzing/EMB/tree/master/jdk_8_maven/cs/rest-gui/genome-nexus#readme',
    'swagger': 'https://raw.githubusercontent.com/WebFuzzing/EMB/refs/heads/master/openapi-swagger/genome-nexus.json',
    'actors': ['Researchers', 'Clinicians'],
    'highLevelGoals': [   'Annotate genetic variants in cancer',
                          'Interpret genetic variants in cancer',
                          'Look up reference gene and protein data'],
    'lowLevelGoals': [   'Search and retrieve variant annotations from a user interface',
                         'Analyze cancer-related mutations using automated tools',
                         'Classify cancer mutations based on their clinical relevance',
                         'Process large genomic datasets in parallel',
                         'Extract and transform mutation data from high-throughput sequencing '
                         'formats (e.g., VCF, BAM)',
                         'Perform mutation quality control and filtering',
                         'Retrieve and harmonize data from multiple genomic databases',
                         'Query integrated genomic databases for relevant mutation information',
                         'Integrate data from multiple compatible genomic sources for easy '
                         'retrieval',
                         'Convert a DNA change to the corresponding protein change',
                         'Convert mutations to amino acid changes for protein function analysis',
                         'Predict the impact of mutations on protein structure using '
                         'bioinformatics tools',
                         'Use prediction tools (e.g., PolyPhen, SIFT) to estimate mutation effects '
                         'on protein function',
                         'Build and apply machine learning models to predict functional impact',
                         'Rank mutations based on their predicted functional impact',
                         'Retrieve mutation frequencies across different population groups',
                         'Provide mutation frequency data_key for specific diseases or conditions',
                         'Retrieve gene function annotations from public databases like Gene '
                         'Ontology (GO)',
                         'Identify pathways and biological processes related to the mutated gene',
                         'Identify how mutations alter protein activity or structure',
                         'Identify mutations with known clinical drug responses or therapeutic '
                         'implications',
                         'Provide actionable insights on mutations based on current clinical '
                         'research',
                         'Retrieve the annotation of a variant by dbSNP id',
                         'Retrieve the annotation of a variant by genomic location',
                         'Retrieve the canonical Ensembl gene for a gene identifier',
                         'Retrieve the Ensembl transcripts of a gene or protein',
                         'Retrieve the external references of an Ensembl gene',
                         'Retrieve a PFAM protein domain',
                         'Retrieve the header of a PDB structure',
                         'Retrieve the post-translational modifications of a transcript']}

SIA_PROJECT_GESTAO_HOSPITAL = {   'name': 'Gestao Hospital',
    'description': 'O objetivo do projeto é criar uma API para organizar um sistema público de '
                   'saúde. O Sistema Único de Saúde (SUS) é um dos maiores e mais complexos '
                   'sistemas de saúde pública do mundo, abrangendo desde o simples atendimento '
                   'para avaliação da pressão arterial, por meio da Atenção Básica, até o '
                   'transplante de órgãos, garantindo acesso integral, universal e gratuito para '
                   'toda a população do país. Com a sua criação, o SUS proporcionou o acesso '
                   'universal ao sistema público de saúde, sem discriminação. A atenção integral à '
                   'saúde, e não somente aos cuidados assistenciais, passou a ser um direito de '
                   'todos os brasileiros, desde a gestação e por toda a vida, com foco na saúde '
                   'com qualidade de vida, visando a prevenção e a promoção da saúde. O objetivo '
                   'desse projeto é criar uma ferramenta para auxiliar o SUS, evitar desperdício e '
                   'potencializar os recursos a partir dos pacientes.',
    'link-readme': 'https://github.com/ValchanOficial/GestaoHospital/blob/master/README.md',
    'swagger': 'https://raw.githubusercontent.com/WebFuzzing/EMB/refs/heads/master/openapi-swagger/gestaohospital-rest.json',
    'actors': ['Hospital Manager', 'Healthcare Staff', 'Hospital Logistics Staff'],
    'highLevelGoals': [   'Manage hospital information',
                          'Manage hospital beds and patients',
                          'Manage hospital stock and blood bank',
                          'Find a suitable hospital for a patient'],
    'lowLevelGoals': [   'Register a new hospital',
                         'Delete a hospital',
                         'Update hospital information',
                         'Find the nearest hospital with available beds',
                         'View hospital information',
                         'View Products and Quantities',
                         'Register a product in stock',
                         'Delete Products',
                         'Update product quantities',
                         'View Info on a Single Product',
                         'Request a product transfer from a nearby hospital',
                         'View patient information',
                         'Register a patient at a hospital',
                         'Update patient information',
                         'Check out a patient from a hospital',
                         'Check available beds',
                         'List patients in a hospital',
                         'Find hospitals near a hospital']}

SIA_PROJECT_LONDON_AMBULANCE_SERVICE = {   'name': 'London Ambulance Service',
    'description': 'The London Ambulance Service dispatches ambulances in emergencies. The key '
                   'goal is to allocate an available ambulance for every call that can reach the '
                   'scene within 11 minutes. The entire dispatch process must not exceed a set '
                   'maximum time limit. The Computer Aided Despatch system analyzes incident forms '
                   'and assigns vehicles. The exact location of moving ambulances must be tracked, '
                   'requiring ambulance staff to follow standard routes and communicate departure '
                   'and destination information so that radio operators and the CAD can accurately '
                   'record the corresponding data.',
    'actors': [   'Computer Aided Despatch (CAD)',
                  'Ambulance Staff',
                  'Radio Operator',
                  'Resource Allocator (RA)'],
    'highLevelGoals': [   'Track moving ambulances continuously',
                          'Allocate an ambulance within 11 minutes of an incident'],
    'lowLevelGoals': [   'Keep exact location data for parked/stationary ambulances',
                         'Ensure ambulances stick to expected standard routes',
                         'Get exact departure and destination data when leaving',
                         'Update location using the departure/destination data',
                         'Get location updates via phone',
                         'Send location/status info via email',
                         'Keep location accurate after reaching a destination',
                         'Staff communicates departure and destination upon leaving',
                         'Operator encodes the departure and destination info',
                         'System records the encoded data']}

SIA_PROJECT_CALCULATOR = {   'name': 'Calculator',
    'description': 'This project develops a smartphone app for a simple calculator. In its initial '
                   'version, it allows the user to do basic calculations. More complex '
                   'calculations will be considered in subsequent versions that are added '
                   'iteratively.',
    'actors': ['User'],
    'highLevelGoals': [   'Perform basic arithmetic calculations',
                          'Interact with the calculator interface'],
    'lowLevelGoals': [   'Add two numbers and obtain their sum',
                         'Subtract one number from another and obtain the difference',
                         'Multiply two numbers and obtain their product',
                         'Divide one number by another and obtain the quotient',
                         'Enter numbers for a calculation',
                         'Select a basic arithmetic operation',
                         'View the result of a calculation']}

SIA_PROJECT_FITNESSTRACKER = {   'name': 'Fitness Tracker',
    'description': 'This project develops a fitness application aimed at helping users track their '
                   'physical activities, monitor progress, set fitness goals, and receive '
                   'personalized workout recommendations. This app is designed to cater to both '
                   'fitness enthusiasts and beginners, providing valuable tools and insights to '
                   'support their fitness journey.',
    'actors': ['User'],
    'highLevelGoals': [   'Track physical activities',
                          'Monitor fitness progress',
                          'Set and manage fitness goals',
                          'Receive personalized workout recommendations',
                          'Use a wearable device with the app',
                          'Share fitness activities and achievements on social media'],
    'lowLevelGoals': [   'Track the duration and distance of daily activities',
                         'Manually log physical activities',
                         'View weekly and monthly progress reports',
                         'Set fitness goals',
                         'Edit fitness goals',
                         'Track goal achievement',
                         'View personalized workout suggestions',
                         'Rate the difficulty of a completed workout',
                         'Connect to wearable devices',
                         'View activity metrics from a wearable device',
                         'Share a completed activity on social media',
                         'Post achievements and milestones on social media',
                         'View the daily activity summary']}

SIA_PROJECT_PLANNINGPOKER = {   'name': 'Planning Poker',
    'description': "This project describes 'Planning Poker' a digital card game designed to help "
                   'empower agile and scrum development teams to collectively set their sprint '
                   'goals. Using anonymous pointing, Planning Poker helps to elicit productive '
                   'conversations across the team to create shared ownership, knowledge, and '
                   'understanding.',
    'actors': ['Moderator', 'Estimator'],
    'highLevelGoals': [   'Set up a planning poker game',
                          'Run an estimation round',
                          'Manage past games',
                          'Manage account',
                          'Participate in estimation sessions'],
    'lowLevelGoals': [   'Create a new game',
                         'Invite estimators to a game',
                         'Add an item for estimation',
                         'Edit an item for estimation',
                         'Delete an item from estimation',
                         'View all items to be estimated in the session',
                         'Select an item for estimation',
                         'Show estimates before all estimators have voted',
                         'Accept the average estimate',
                         'Enter the agreed-upon estimate',
                         'Re-estimate a previously estimated story',
                         'Import stories from a spreadsheet',
                         'Copy and paste stories from a spreadsheet',
                         'Browse previous games',
                         'View a game transcript',
                         'Export a game transcript as HTML',
                         'Export a game transcript as CSV',
                         'Delete a game',
                         'Create an account',
                         'Log in to an account',
                         'Change account details',
                         'Delete an account',
                         'Request a password reminder by email',
                         'Select an estimation scale',
                         'Join a game',
                         'View the item being estimated',
                         'View who gave each estimate',
                         'View which estimators have already given an estimate',
                         'View prior estimates for the current story',
                         'Change an estimate before all cards are revealed',
                         'Start a countdown timer',
                         'Reset the countdown timer after an estimation round',
                         'View stories and estimates from previous rounds']}

SIA_PROJECT_RECYCLING = {   'name': 'Recycling',
    'description': 'The software project will develop a user-friendly web application that allows '
                   'users to discover and manage recycling facilities. Key features include '
                   'searching for nearby centers by zip code, viewing operational hours, accessing '
                   'maps of recycling locations, and receiving notifications about events. '
                   'Administrators will manage facility information and user feedback, ensuring '
                   'data security while enabling communication between facility representatives '
                   'and site admins to improve community recycling efforts.',
    'actors': ['User', 'Admin', 'Superuser', 'Recycling Facility Representative'],
    'highLevelGoals': [   'Manage user accounts and preferences',
                          'Discover and navigate recycling facilities',
                          'Schedule and optimize recycling visits',
                          'Administer recycling facility information',
                          'Handle user feedback and communication'],
    'lowLevelGoals': [   'Create a user account',
                         'Add recycling facilities to favorites',
                         'Link an email account to a profile',
                         'Subscribe to notifications about new events',
                         'View user documentation',
                         'Search recycling facilities by zip code',
                         'View recycling centers on a map',
                         'View public recycling bins',
                         'View special waste drop-off sites',
                         'View safe disposal events',
                         'Open a facility address in Google Maps',
                         'Browse environment-friendly recycling facilities',
                         'Choose a flexible pickup time',
                         'Get recycling center recommendations based on a weekly schedule',
                         'Filter recycling facilities by type of recyclable waste',
                         'Add recycling facility information',
                         'Remove recycling facility information',
                         'Onboard recycling centers on the platform',
                         'Update recycling facility information',
                         'Update facility information and accepted material types',
                         'Contact administrators',
                         'Read user feedback and complaints',
                         'Reply to user questions',
                         'Communicate directly with recycling facilities',
                         'Communicate directly with the site admin']}

# Canonical dataset registry shared by the execution and evaluation notebooks.
_REQUIRED_GROUNDTRUTH_FIELDS = {
    "name",
    "description",
    "actors",
    "highLevelGoals",
    "lowLevelGoals",
}


def _discover_groundtruths(namespace):
    projects = []
    for variable_name, value in namespace.items():
        if not variable_name.startswith("SIA_PROJECT_"):
            continue
        if not isinstance(value, dict):
            raise TypeError(f"{variable_name} must be a dictionary")

        missing_fields = _REQUIRED_GROUNDTRUTH_FIELDS - value.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"{variable_name} is missing fields: {missing}")
        projects.append(value)

    if not projects:
        raise RuntimeError("No SIA_PROJECT_* ground truths were found")

    names = [project["name"] for project in projects]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(
            "Ground-truth project names must be unique: "
            + ", ".join(duplicate_names)
        )

    return projects


ALL_GROUNDTRUTHS = _discover_groundtruths(globals())

GROUNDTRUTH_BY_NAME = {
    groundtruth["name"]: groundtruth
    for groundtruth in ALL_GROUNDTRUTHS
}
