
SIA_PROJECT_25_26 = {
    "name" : "SIA Project 25 26",
    "description": """
    In the Municipality of Turin, a system called Participium is being developed — an information 
system that enables citizen participation in the management of urban environments. It allows 
citizens to interact with the public administration by reporting inconveniences and malfunctions 
found in the city (e.g., potholes in the asphalt, sidewalks with architectural barriers, garbage in the 
streets, broken streetlights, etc.). An example of a similar information system is IRIS in Venice: 
https://iris.sad.ve.it/. The system, once developed, will be made available as open-source 
software to all Italian public administrations through the portal https://developers.italia.it/.

A. REPORTS 
Citizens can submit reports only if they have registered in the system with a username, first name, 
and last name. Once the registration is received, the user gets an email with a confirmation link. 
The registration becomes valid, and the user can use the system only after confirming through 
that link. Once registered, a citizen can submit reports by selecting a point on the map of Turin 
(which will be saved with latitude and longitude values) and filling in a problem form with the 
following mandatory fields: title, textual description, category (chosen from a predefined list). It is 
also mandatory to attach one or more photos (up to 3 per report, with each photo stored with its 
file path on the server). The possible problem categories are: 
• Waterworks – Drinking water 
• Architectural Barriers 
• Sewerage 
• Public Lighting 
• Waste 
• Road Signs and Traffic Lights 
• Roads and Urban Furniture 
• Public Green Areas and Playgrounds 
• Other 
After entering all the information and the pictures, the system asks the citizen whether they want 
the report to be anonymous (the name will not appear in the public list of all reports).  

B. REPORT LIFECYCLE 
Once submitted, the report is in the “Pending Approval” state until the Organization Office of the 
Municipality of Turin performs a preliminary review of citizen reports, marking them as accepted 
or rejected. The possible states for a report are: Pending Approval, In Progress, Work in Progress, 
Suspended, Rejected. After approval, accepted reports move to the “In Progress” state, at which 
point they are assigned to the technical office responsible, based on the problem category. Once 
the intervention is planned, the state changes to “Work in Progress”, indicating that the issue 
resolution has started. In some cases, for organizational or technical reasons, the report can be 
set to “Suspended”, awaiting further evaluation or resources. When the problem is solved, the 
technical office updates the status and closes the report. In case of rejection, a written explanation 
from the Organization Office is mandatory (see next section).  
If the intervention must be carried out by maintenance personnel from an external company (for 
example, Enel X for Public Lighting, or company Y for specific reports based on their content), 
two cases are possible: - Case 1: The company has access to Participium. In this case, the technical office assigns the 
report to users from the corresponding company. External maintenance personnel can move the 
report from Assigned to In Progress. Staff from the technical office and external maintenance 
workers can exchange information and comments through the report, but these are not visible to 
the reporting citizen (nor to other citizens). Once the work is completed, the external maintainer 
can mark the report as resolved. 
Note: Automatic assignment of all reports in a certain category to the external company 
(bypassing the initial review by the public relations officer of the municipality) is possible only if 
previously configured by the municipal administrator. - Case 2: The company does NOT have access to Participium. In this case, the external company 
updates the technical office outside Participium, and once the problems are solved, a staff 
member from the technical office will manually move the report to the Resolved state. 


C. CITIZEN UPDATES 
To strengthen trust between citizens and institutions, the citizen can receive updates regarding 
their own reports through various channels. First, at every state change, the citizen receives a 
notification on the platform with the corresponding update. Furthermore, municipal operators 
updating reports can send the reporting citizen a message through the platform, to which it can 
reply. The system must allow this functionality to be accessible by external chatbots as well. Each 
time the citizen receives a notification on the platform, they also get an email (this option can be 
disabled in the user settings panel, where they can also upload a personal photo). 
Moreover, after the approval phase, accepted reports immediately become visible on the 
Participium portal: they appear both on an interactive map of Turin, geolocated based on the 
citizen’s selected point, and in a summary table that allows filtering and sorting reports by 
category, status, or date. In both views, the reporter’s name (“anonymous” if that option was 
chosen) and the report title are displayed. Clicking on the title opens a page with the description.  
Even non-registered users can view both the maps and the summary table. Once logged into 
Participium, citizens can follow other citizens’ open reports and receive notifications (following the 
same rules as for their own reports). 


D. STATISTICS 
The system allows viewing public and private statistics. Public statistics are visible in a dedicated 
section of the website and concern the number of reports per category, and trends over day, week, 
or month. They are also visible to non-registered users. In the private section, accessible only to 
administrators, it is possible to view, in addition to the public statistics, charts and tables. 
    """,

    "actors": [
        "Visitor",
        "Citizen",
        "Technical Office Staff",
        "Organizational Office Staff",
        "Municipal Administrator",
        "External Maintenance Staff"
    ],

    "highLevelGoals": [
        "View public statistics",
        "Register an account",
        "Submit a report",
        "Follow reports",
        "Exchange messages",
        "Manage report lifecycle",
        "View private statistics",
        "Configure automatic report assignment",
        "Manage account information",
        "Browse public reports",
        "Receive updates about reports"
    ],

    "lowLevelGoals": [
        "Activate account",
        "Submit account registration",
        "Associate a report with a specific location on the map",
        "Select anonymity for a report",
        "Submit a report",
        "Send a message",
        "Activate email notifications",
        "Deactivate email notifications",
        "Upload a profile photo",
        "Activate auto-assignment for a category",
        "Deactivate auto-assignment for a category",
        "Approve a report",
        "Reject a report specifying the rejection reason",
        "Add a comment related to a report",
        "Assign a report to an external company",
        "Follow a report",
        "View the details of a report",
        "Log in",
        "Attach photos to a report",
        "Assign a report to the responsible technical office",
        "Set a report status to In Progress",
        "Set a report status to Work in Progress",
        "Suspend a report",
        "Mark a report as resolved",
        "Close a resolved report",
        "Receive a platform notification when a report status changes",
        "Receive an email notification when a report status changes",
        "Receive notifications about a followed report",
        "Reply to a message from a municipal operator",
        "Make report messaging accessible to external chatbots",
        "View accepted reports on the interactive map",
        "View accepted reports in the summary table",
        "Display the reporter's name or an anonymous label",
        "Filter reports by category, status, or date",
        "Sort reports by category, status, or date",
        "View public report statistics",
        "View private report statistics"
    ]
}

SIA_PROJECT_24_25 = {
    "name": "Assegno Unico Universale - SIA Project 24 25",

    "description": """
    The system supports the online management of applications for the Italian
    Assegno Unico Universale (AUU), a financial benefit provided by INPS to
    families with dependent children. The considered scenario focuses exclusively
    on applications submitted directly by interested users through the INPS website,
    excluding applications submitted by delegated patronage institutions, mobile
    applications, and call-center interactions.

    An authenticated applicant can submit an AUU application as a parent,
    foster parent, or legal guardian of a child. During the first phase of the
    application, the applicant provides information about each dependent child,
    including the child's fiscal code, dependency status, disability status and,
    when applicable, disability level. Depending on the applicant type, additional
    information concerning family composition, exclusive custody, the other
    parent's fiscal code, and the requested allowance distribution between parents
    must also be provided.

    When a child's fiscal code is entered, the system verifies it in real time.
    If the child is already associated with another active application, the
    application flow is blocked and an explanatory warning is shown. The applicant
    may also specify adult dependent children who do not directly qualify for AUU
    but contribute to determining household composition. Duplicate fiscal codes for
    adult children are detected and reported.

    The amount of the allowance depends on the applicant's economic situation.
    If INPS does not already possess a valid ISEE when the application is submitted,
    the minimum allowance established by regulation is used. If a valid ISEE becomes
    available later, INPS automatically recalculates the allowance without requiring
    the applicant to modify the application.

    In the payment phase, the applicant specifies the payment method. If the
    allowance is to be divided equally between both parents, the payment information
    of the other parent must also be provided. Supported payment methods include
    credit to a SEPA IBAN account owned by the beneficiary, domiciled payment at a
    post office, and credit to an eligible prepaid card.

    Before submitting the application, the applicant must read and accept the
    declarations of responsibility and the GDPR privacy notice. The system then
    presents a summary of the entered information so that it can be reviewed.
    If validation errors are detected, the application is not submitted and the
    applicant must correct the erroneous information. If validation succeeds, the
    application is submitted and a receipt can be downloaded.

    An application may be interrupted without saving or saved as a draft before
    submission. A saved draft can later be consulted and completed. When resuming
    a draft, the applicant may either continue from the previously saved information
    or discard it and restart the application from the beginning.

    Once submitted, the application is evaluated by authorized INPS staff.
    The application may be accepted, rejected because the applicant is not entitled
    to the benefit, or suspended when additional documentation is required.
    The applicant is notified of the outcome. An accepted application remains valid
    for twelve months and the corresponding allowance is paid automatically every
    month.

    When an application is suspended, the INPS operator specifies which additional
    documents are required. The applicant can consult the application, upload the
    requested attachments, reconfirm the application data, and again accept the
    responsibility declarations and privacy notice.

    The system also allows an application originally submitted by one parent to be
    completed by the other parent. The other parent can consult the application and
    confirm or modify the proposed allowance distribution. If the applicant requested
    the entire allowance, the other parent may accept the 100 percent allocation or
    request a 50 percent distribution. When payment to the other parent is required,
    that parent provides their own payment information.

    Applicants can also add one or more children to an existing application.
    Each new child's fiscal code is checked in real time to ensure that the child
    is not already associated with another active application. After adding the
    child, existing payment and declaration data can be confirmed or modified.

    Finally, the system provides supporting functions that allow citizens to consult
    previously submitted applications, consult general information about the
    Assegno Unico Universale, and simulate the expected monthly allowance amount.
    """,

    "actors": [
        "Applicant",
        "Other Parent",
        "INPS Employee"
    ],

    "highLevelGoals": [
        "Submit and manage an AUU application",
        "Complete an AUU application as the other parent",
        "Evaluate and manage submitted AUU applications",
        "Access AUU information and decision-support services"
    ],

    "lowLevelGoals": [
        "Specify personal and dependent child information",
        "Specify custody and household information",
        "Specify the requested allowance distribution",
        "Specify applicant payment information",
        "Specify the other parent's payment information as the applicant",
        "Specify information about adult dependent children",
        "Review application data before submission",
        "Submit the AUU application",
        "Download the application receipt",
        "Save an incomplete application as a draft",
        "Resume and complete a saved application",
        "Consult an AUU application",
        "Upload additional documents requested by INPS",
        "Add a child to an existing application",
        "Consult general information about the AUU",
        "Simulate the monthly AUU amount",
        "Confirm or modify the allowance distribution as the other parent",
        "Specify personal payment information as the other parent",
        "Accept an AUU application",
        "Reject an AUU application",
        "Suspend an AUU application pending additional documentation",
        "Verify a child's fiscal code in real time",
        "Detect a child already associated with another active application",
        "Detect duplicate fiscal codes for adult dependent children",
        "Use the minimum allowance when a valid ISEE is unavailable",
        "Automatically recalculate the allowance when a valid ISEE becomes available",
        "Accept the declarations of responsibility and the GDPR privacy notice",
        "Validate application data before submission",
        "Correct invalid application information",
        "Interrupt an application without saving it",
        "Discard a saved draft and restart the application",
        "Notify the applicant of the application outcome",
        "Pay an accepted allowance automatically every month",
        "Specify the additional documents required from the applicant",
        "Reconfirm application data after a suspension"
    ]
}

SIA_PROJECT_23_24 = {
    "name": "La Reine Marlene - SIA Project 23 24",

    "description": """
    La Reine Marlene is a company specialized in the sale of home materials
    and products. The company is redesigning its information system in order
    to improve operational efficiency and customer service.

    The system supports interactions with a large network of suppliers.
    Suppliers must register in the system by providing credentials and company
    information, including VAT number, address, and an IBAN for payments.
    Registered suppliers can submit proposals for their products by specifying
    the product category, selling price to La Reine Marlene, availability,
    photos, description, and guaranteed delivery time.

    After a product proposal has been submitted, the supplier cannot modify it
    until a response is received. The administration department is notified
    whenever a new proposal is submitted and evaluates it by choosing among
    three possible outcomes: accepted, revisions requested, or rejected.

    If a proposal is accepted, a separate order-management process starts and
    the retail price to be displayed in the store is added to the order
    information. If revisions are requested, the administration department
    must explain them in a comment. The supplier is notified and has five
    working days to modify the proposal, add response comments, and resubmit it.
    If the supplier does not modify the proposal within the deadline, the
    proposal is removed from the system and the supplier is notified.

    Administrators receive a reminder every 48 hours for proposals that are
    still waiting to be evaluated. If a proposal is not evaluated within ten
    working days, the system automatically rejects it and notifies the supplier.

    Once ordered products are shipped and arrive at one of the La Reine Marlene
    stores, the goods-receiving staff verifies each product. Since the company
    does not use warehouses, every available product is directly exposed in the
    store. Products are catalogued according to category, available quantity,
    and store location.

    During goods reception, the operator checks each delivered product. If no
    problem is found, the product is confirmed and the system proposes a list
    of possible store locations. The operator selects a location and the product
    is passed to the staff responsible for placing it in the store and printing
    the corresponding price. If a problem is detected, the operator reports it
    by adding a comment and the product is not admitted into the store.

    At the end of the receiving procedure, the order department processes the
    result. If all products have been accepted, the order is completed and the
    supplier is paid. Otherwise, the order is updated and remains open while
    waiting for a later delivery that completes it. In both cases, specific
    notifications are sent to the supplier.

    The system tracks inventory levels in real time. When a product reaches a
    predefined critical reorder threshold, the order department receives an
    automatic notification. Administrators can configure automatic reorder and
    replenishment actions for selected products.

    In case of automatic reordering, an order is sent to a predefined supplier,
    who has two working days to confirm it. If no critical threshold and
    automatic reorder rule are defined, the order department manages the new
    order through a separate process.

    Customers can purchase products either in physical stores or through the
    online store. Registered online customers can browse the digital catalogue,
    select products and quantities, and complete an order. The system
    automatically calculates the order total, including taxes and shipping costs.

    Customers can pay using Visa or Mastercard credit cards, PayPal, Satispay,
    the La Reine Marlene purchase card, or an immediate online bank transfer
    through supported banks.

    Once an online order has been paid, the system tracks the delivery in real
    time. Customers receive a link by email that allows them to follow the
    shipment status. Customers who have provided consent to use their mobile
    phone number also receive notifications when estimated delivery times are
    updated. Once the product is delivered, the order is marked as completed.

    Deliveries are performed by partner shipping companies. Each online purchase
    is randomly assigned to a shipping provider. The same online shopping
    procedures are available both through the website and through the dedicated
    application.

    The information system also provides customer support. Customers can submit
    questions, assistance requests, and complaints by opening support tickets.
    Each support ticket is automatically assigned to a customer-service agent.
    The agent is notified and has 24 hours to respond. If no response is provided
    within that time, the ticket is reassigned to another operator and service
    statistics are updated.

    When the customer receives a response, the customer is notified and has five
    working days to reply. If no reply is received within that period, the ticket
    is automatically considered closed.

    Finally, the system collects data regarding orders, sales, inventory, and
    customer support. These data are used by company management to analyze
    business performance and support strategic decision making.
    """,

    "actors": [
        "Supplier",
        "Administration Department",
        "Goods Receiving Staff",
        "Order Department",
        "Customer",
        "Point-of-Sale System",
        "Shipping System",
        "Messaging Gateway",
        "Customer Service Agent",
        "Company Management"
    ],

    "highLevelGoals": [
        "Supply products to La Reine Marlene",
        "Manage product proposals",
        "Manage incoming goods",
        "Manage supplier orders",
        "Purchase products",
        "Request customer support",
        "Provide customer support",
        "Analyze company performance",
        "Manage product inventory and replenishment",
        "Manage online order delivery"
    ],

    "lowLevelGoals": [
        "Submit a product proposal",
        "Accept a product proposal",
        "Request revisions to a product proposal",
        "Reject a product proposal",
        "Prepare a supplier order",
        "Send a supplier order",
        "Confirm an automatically generated order",
        "Verify delivered goods",
        "Report problems with delivered products",
        "Select a store location for accepted products",
        "Complete a supplier order",
        "Update an incomplete supplier order",
        "Pay a supplier order",
        "Configure automatic product reordering",
        "Monitor product stock levels",
        "Browse the product catalogue",
        "Select products and quantities",
        "Purchase products online",
        "Calculate the order total",
        "Pay an online order",
        "Track shipment status",
        "Open a customer support ticket",
        "View responses to a support ticket",
        "Respond to a customer support ticket",
        "View business statistics",
        "Register as a supplier",
        "Notify the administration department of a new product proposal",
        "Set the retail price for an accepted product proposal",
        "Explain requested revisions in a comment",
        "Notify the supplier of a product proposal decision",
        "Modify and resubmit a product proposal within five working days",
        "Remove an unrevised product proposal after its deadline",
        "Remind administrators about unevaluated product proposals",
        "Automatically reject an overdue product proposal",
        "Confirm accepted products for admission into the store",
        "Place accepted products in the selected store location",
        "Print the price of an accepted product",
        "Notify the order department when stock reaches a critical threshold",
        "Automatically send a reorder to a predefined supplier",
        "Assign an online purchase to a shipping provider",
        "Send a shipment tracking link to the customer",
        "Notify the customer of updated delivery times",
        "Mark an online order as completed after delivery",
        "Automatically assign a customer support ticket to an agent",
        "Notify the assigned customer service agent",
        "Reassign an unanswered customer support ticket",
        "Update customer service statistics",
        "Notify the customer of a support response",
        "Reply to a customer service response",
        "Automatically close a customer support ticket after the reply deadline"
    ]
}

SIA_PROJECT_22_23 = {
    "name": "Ethical Purchasing Group - SIA Project 22 23",

    "description": """
    The system supports the management of an Ethical Purchasing Group (EPS),
    known in Italian as a Gruppo di Acquisto Solidale (GAS), associated with a
    km-zero fruit and vegetable store. The purpose of the system is to connect
    groups of customers with local farmers and store employees, supporting a
    conscious purchasing process based on knowledge of product origin, quality,
    characteristics, availability, and price.

    The operating cycle of the purchasing group is organized on a weekly basis.
    Farmers provide estimates of the products they can supply between Friday
    at noon and Saturday morning. For each available product, farmers specify
    the product type, quantity, price, and optionally additional crop
    characteristics such as size, quality, or ripeness.

    The information system maintains a catalogue of products containing
    descriptions, photographs, the unit of measurement used for sale, and the
    minimum and maximum quantities that can be ordered. Each availability
    entered by a farmer refers to one of the products contained in the catalogue.

    Customers can browse the offers provided by the different farmers and place
    orders by selecting products and specifying the desired quantities. Customers
    are representatives of purchasing groups, such as families or groups of
    people, and only the registered representative of each group can submit an
    order.

    Orders can be entered directly by customers through the web system or by
    store employees on behalf of customers. Before an order can be confirmed,
    the system checks that the quantities selected respect the minimum order
    constraints defined for each product. If the minimum quantities are not met,
    the customer must modify the order. After completion, the system sends an
    order summary to the customer by email.

    Customer orders initially remain in an entered state until farmers confirm
    the actual quantities available by Monday morning. If the available
    quantities are sufficient to satisfy all orders, the orders are confirmed.
    When the available quantity of a product is insufficient, the system applies
    precedence rules based on customer priority and removes unavailable
    quantities from lower-priority orders.

    Orders modified because of limited product availability are placed in a
    modified state and must be reviewed and explicitly accepted by customers.
    If every product in an order becomes unavailable, the order is cancelled.

    After farmers confirm product availability, the system automatically
    performs payment for the quantities actually included in the order.
    Payments are made through a personal virtual wallet associated with each
    customer. Customers can recharge their wallet through the store at any time.

    If a customer's wallet balance is insufficient immediately after order
    placement, the customer receives an email warning, but the order remains
    valid. If the balance is insufficient when the automatic payment is executed,
    the customer receives an SMS and has until Monday evening to recharge the
    wallet. If the required balance is still unavailable at that time, the order
    is cancelled.

    Once payment has been successfully completed, customers can reserve a time
    slot for collecting their products from the store. Store employees can view
    the pickup schedule at the beginning of their shifts and prepare the packages
    corresponding to customer orders. When a customer arrives, the employee
    confirms the collection of the order.

    If a customer fails to collect the order at the scheduled time, the system
    sends an SMS reminder. If the order has not been collected by Friday, the
    order is marked as forfeited and the customer's fair-play score is updated.

    Farmers deliver their products to the store by Tuesday evening. Store
    employees confirm the receipt of the delivered quantities. If the goods
    have not arrived by 4 p.m. on Tuesday, the system sends an SMS reminder.
    If the goods have still not been delivered by 8 p.m., the related customer
    orders are updated, the missing amount is credited back to the affected
    customers with an additional 20 percent bonus, and the customers are
    informed of the delivery problem.

    The system maintains a fair-play score for each customer. The score represents
    the reliability of the customer with respect to previous orders and is used
    as a precedence criterion when product availability is insufficient.
    The score is computed from the number of completed orders, reminders received,
    forfeited pickups, and modified orders. Customers with higher fair-play scores
    are given precedence over customers with lower scores when limited quantities
    must be allocated among competing orders.
    """,

    "actors": [
        "Customer",
        "Farmer",
        "Store Employee"
    ],

    "highLevelGoals": [
        "Manage customer orders",
        "Manage product supply and provisioning",
        "Manage customer payments",
        "Manage order pickup",
        "Manage customer fair-play scores"
    ],

    "lowLevelGoals": [
        "Place an order",
        "Accept a modified order",
        "Book an order pickup",
        "Enter an order on behalf of a customer",
        "View order details",
        "Confirm an order pickup",
        "Recharge a customer wallet",
        "Confirm product delivery",
        "Enter product availability",
        "Confirm product availability",
        "Browse offers from farmers",
        "Select products and quantities for an order",
        "Validate minimum order quantities",
        "Modify an order that does not meet minimum quantity constraints",
        "Send an order summary to the customer by email",
        "Allocate limited product quantities according to customer priority",
        "Update orders according to confirmed product availability",
        "Cancel an order when all its products are unavailable",
        "Automatically pay an order using the customer wallet",
        "Notify a customer of an insufficient wallet balance",
        "Cancel an order when the wallet balance remains insufficient",
        "View the order pickup schedule",
        "Prepare packages for customer orders",
        "Send a reminder for a missed order pickup",
        "Mark an uncollected order as forfeited",
        "Send a reminder for a missing product delivery",
        "Refund missing product amounts with a twenty percent bonus",
        "Notify customers of a product delivery problem",
        "Calculate a customer fair-play score",
        "Update a customer fair-play score"
    ]
}

SIA_PROJECT_21_22 = {
    "name": "Event Organization Portal - SIA Project 21 22",

    "description": """
    The system is a web portal designed to support people who want to organize
    an event by bringing together, in a single platform, information about event
    guests and the various services required for the event.

    The platform supports three main types of users: event organizers, service
    providers, and invited guests.

    Service providers register on the platform by providing authentication
    credentials, personal information, and company details, including VAT
    number, company address, and accepted payment methods.

    Once registered, providers can publish the services offered by their company.
    Each service belongs to a category, such as catering, hairdressing, florist
    services, or transportation. For each service, the provider can configure a
    virtual showcase containing demonstration photos, prices, and the time slots
    during which the service is available.

    Event organizers register by providing credentials, personal information,
    residence information, their fiscal code, and an IBAN that can be used for
    service payments.

    Through the portal, organizers can create an event by specifying the event
    type, date, location, and the planned budget. During event creation, they can
    also select from a predefined list the categories of services for which they
    want to receive support.

    Organizers can add invited guests to the event by entering their personal
    information, email address, and guest category, such as relative, friend, or
    colleague. For each invited guest, the system generates an invitation code
    that is sent by email and is used by the guest as a password to access the
    platform.

    Each event has an associated blog where both organizers and invited guests
    can publish information about the event, upload photos or videos, create new
    posts, and comment on existing posts.

    After creating an event, the organizer can browse the service showcases
    provided by suppliers for the selected service categories. The organizer can
    then request an appointment with a provider by choosing a time compatible
    with the provider's availability and specifying appointment details and any
    special requests.

    Providers can view appointment requests through their interface and decide
    whether to approve them. Approved appointments are automatically displayed
    in a calendar associated with the event, which the organizer can consult to
    view all confirmed service appointments.

    After an appointment between the organizer and the provider has taken place,
    the provider prepares a quotation for the agreed service and uploads the
    corresponding amount to the portal, associating it with the completed
    appointment.

    Once a quotation has been uploaded, the organizer can accept or reject it.
    If the quotation is accepted, its total amount is deducted from the remaining
    event budget.

    When accepting a quotation, the organizer selects both the payment schedule,
    which can consist of installments, deposit plus final balance, or full payment,
    and the payment method, which can be online or on-site.

    If payment is made on-site, the provider confirms the received amount through
    the portal. If payment is divided into multiple installments, the provider
    confirms each received installment separately.

    At any time, the organizer can consult the history of contracts associated
    with the event, including their status and cost, and can also view the
    remaining event budget.
    """,

    "actors": [
        "Organizer",
        "Service Provider",
        "Guest"
    ],

    "highLevelGoals": [
        "Manage event services",
        "Manage an event",
        "Manage the event blog"
    ],

    "lowLevelGoals": [
        "Register as a service provider",
        "Add offered services",
        "Confirm an on-site payment",
        "Register as an organizer",
        "Create an event",
        "Add invited guests",
        "Select services for an event",
        "View appointment calendar",
        "View the remaining event budget",
        "View contract history",
        "Create a blog post as an organizer",
        "Create a blog post as a guest",
        "Configure a virtual service showcase",
        "Upload demonstration photos for a service",
        "Specify the price of a service",
        "Specify service availability time slots",
        "Browse service showcases",
        "Request an appointment with a service provider",
        "Approve an appointment request",
        "Create and upload a quotation for a completed appointment",
        "Accept a quotation",
        "Reject a quotation",
        "Deduct an accepted quotation from the remaining event budget",
        "Select a payment schedule",
        "Select a payment method",
        "Confirm a received payment installment",
        "Generate and email an invitation code to a guest",
        "Access the platform using an invitation code",
        "Upload photos or videos to the event blog",
        "Comment on an event blog post"
    ]
}

SIA_PROJECT_GENOME_NEXUS = {
    "name": "Genome Nexus",
    "link-readme": "https://github.com/WebFuzzing/EMB/tree/master/jdk_8_maven/cs/rest-gui/genome-nexus#readme",
    "swagger": "https://raw.githubusercontent.com/WebFuzzing/EMB/refs/heads/master/openapi-swagger/genome-nexus.json",
    "description": """
    Genome Nexus is a web service that supports the annotation and interpretation
    of genetic variants. The system allows researchers and clinicians to retrieve
    information about variants from multiple genomic and clinical resources,
    including mutation annotations, clinical significance, disease associations,
    mutation frequencies, and gene-related information.

    The system supports the analysis of cancer-related mutations and high-throughput
    genomic datasets. It can map variants to reference genome assemblies, convert
    DNA-level changes into corresponding protein-level changes, and provide
    functional predictions about the possible impact of mutations on proteins.

    Genome Nexus also helps users interpret the biological and clinical relevance
    of variants by integrating information from public databases, prediction tools,
    and clinical knowledge sources. Its goal is to provide a fast, automated, and
    centralized resource for variant annotation and interpretation.
    """,
    "actors": [
        "Researcher",
        "Clinician"
    ],
    "highLevelGoals": [
        "Provide fast and automated annotation of genetic variants",
        "Enable high-throughput interpretation of genetic variants",
        "Integrate information from existing genomic resources",
        "Convert DNA changes to protein changes",
        "Predict functional effects of protein mutations",
        "Provide information about mutation frequencies",
        "Offer insights into gene function",
        "Detail variant effects",
        "Highlight clinical actionability of variants"
    ],
    "lowLevelGoals": [
        "Retrieve genetic variant data_key from multiple databases (e.g., dbSNP, ClinVar, COSMIC)",
        "Search and retrieve variant annotations from a user interface",
        "Annotate variants with clinical significance, mutation types, and related diseases",
        "Map genetic data_key to genome assemblies (e.g., GRCh38, hg19)",
        "Update variant information regularly from authoritative sources",
        "Analyze cancer-related mutations using automated tools",
        "Integrate gene expression data_key for cancer variant interpretation",
        "Identify cancer-related mutations linked to specific pathways",
        "Interpret large-scale cancer mutation datasets automatically",
        "Classify cancer mutations based on clinical relevance",
        "Process large genomic datasets in parallel",
        "Extract and transform mutation data_key from high-throughput sequencing formats (e.g., VCF, BAM)",
        "Perform mutation quality control and filtering",
        "Fetch and harmonize data_key from various genomic databases",
        "Query integrated genomic databases for relevant mutation information",
        "Integrate multiple data_key sources with compatible formats for easy retrieval",
        "Map genetic mutations to corresponding protein-coding effects",
        "Convert mutations to amino acid changes for protein function analysis",
        "Predict the impact of mutations on protein structure using bioinformatics tools",
        "Use prediction tools (e.g., PolyPhen, SIFT) to estimate mutation effects on protein function",
        "Build and apply machine learning models for functional impact prediction",
        "Rank mutations based on predicted severity of functional impact",
        "Calculate mutation frequencies across various population groups",
        "Generate visual representations of mutation frequencies (e.g., histograms, pie charts)",
        "Provide mutation frequency data_key for specific diseases or conditions",
        "Retrieve gene function annotations from public databases like Gene Ontology (GO)",
        "Identify pathways and biological processes related to the mutated gene",
        "Link genetic variants to specific diseases or phenotypes based on annotations",
        "Predict the effects of mutations on protein folding and stability",
        "Identify how mutations alter protein activity or structure",
        "Evaluate the impact of mutations on protein-protein interactions",
        "Link genetic mutations to clinical guidelines or treatment protocols",
        "Identify mutations with known clinical drug responses or therapeutic implications",
        "Provide actionable insights on mutations based on current clinical research"
    ]
}

SIA_PROJECT_GESTAO_HOSPITAL = {
    "name": "Gestao Hospital",
    "link-readme": "https://github.com/ValchanOficial/GestaoHospital/blob/master/README.md",
    "swagger": "https://raw.githubusercontent.com/WebFuzzing/EMB/refs/heads/master/openapi-swagger/gestaohospital-rest.json",
    "description": """
    Gestao Hospital is a hospital management system that supports the administration
    of hospitals, patients, beds, products, and blood bank resources. The system
    allows administrators and hospital managers to register, update, delete, and
    monitor hospitals, while healthcare staff can manage operational information
    related to patients, appointments, products, and medical resources.

    Patients can use the system to search for hospitals, obtain information about
    specific hospitals, and receive recommendations about the nearest hospital.
    Hospital staff can register patients, consult patient information and medical
    history, manage check-in related information, and maintain notes about
    treatments.

    The system also supports hospital logistics by allowing authorized staff to
    manage products, quantities, requests, and blood samples. Its overall goal is
    to centralize hospital administration and improve the coordination of clinical
    and logistical activities.
    """,
    "actors": [
        "Hospital Manager",
        "Healthcare Staff",
        "Administrator",
        "Patient",
        "Hospital Logistics Staff"
    ],
    "highLevelGoals": [
        "Allow administrators and hospital managers to manage a hospital",
        "Allow hospital managers and healthcare staff to manage hospital beds and patients",
        "Allow healthcare staff to manage products and the blood bank",
        "Allow patients to look for hospitals"
    ],
    "lowLevelGoals": [
        "Allow Registration of a New Hospital",
        "Allow Deletion of a Hospital",
        "Allow Modification of a Hospital",
        "Allow administrators to access statistics and manage indicators",
        "Enable hospital staff to manage appointment schedules",
        "Recommend Nearest Hospital",
        "Return Information on a Hospital",
        "View Products and Quantities",
        "Allow logistic staff to Register Products",
        "Delete Products",
        "Allow logistic staff to change product quantities",
        "View Info on a Single Product",
        "Request a Product",
        "Allow searching for blood samples",
        "Enable healthcare staff to view patient info and their medical history",
        "Register a Patient at a Hospital, entering personal information and contact info",
        "Allow patients to confirm their arrival at the hospital online or in presence",
        "Show estimated check in times for patient arriving at the hospital",
        "Allow healthcare staff to save notes regarding patients and their treatment in the system",
        "Change Patient Info and medical history"
    ]
}

SIA_PROJECT_LONDON_AMBULANCE_SERVICE = {
    "name": "London Ambulance Service",
    "description": """
    The London Ambulance Service dispatches ambulances in emergencies. The key
    goal is to allocate an available ambulance for every call that can reach the
    scene within 11 minutes. The entire dispatch process must not exceed a set
    maximum time limit.

    To make this work, the Computer Aided Despatch system analyzes incident forms
    and assigns vehicles. It is vital to track the exact location of moving
    ambulances. This requires ambulance staff to follow standard routes and
    correctly communicate departure and destination information, allowing radio
    operators and the CAD system to record data accurately.
    """,
    "actors": [
        "Computer Aided Despatch (CAD)",
        "Ambulance Staff",
        "Radio Operator",
        "Resource Allocator (RA)"
    ],
    "highLevelGoals": [
        "Track moving ambulances continuously",
        "Allocate an ambulance within 11 minutes of an incident"
    ],
    "lowLevelGoals": [
        "Keep exact location data for parked/stationary ambulances",
        "Ensure ambulances stick to expected standard routes",
        "Get exact departure and destination data when leaving",
        "Update location using the departure/destination data",
        "Get location updates via phone",
        "Send location/status info via email",
        "Keep location accurate after reaching a destination",
        "Staff communicates departure and destination upon leaving",
        "Operator encodes the departure and destination info",
        "System records the encoded data"
    ]
}


# Canonical dataset registry shared by the execution and evaluation notebooks.
# Every SIA_PROJECT_* dictionary defined above is discovered automatically, so
# adding a project does not require updating a second, manually maintained list.
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
